import tensorflow as tf
import numpy as np
import argparse
import os
from transformer.dataset import get_dataset, preprocess_sentence


os.environ["CUDA_VISIBLE_DEVICES"] = " "


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, hparams, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = tf.cast(hparams.d_model, dtype=tf.float32)
    self.warmup_steps = warmup_steps


cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, hparams.max_length - 1))
    loss = cross_entropy(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)


def accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, hparams.max_length - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


class Position(tf.keras.layers.Layer):

    """叠加位置信息"""

    def __init__(self, hparam):
        super(Position, self).__init__()
        self.hparam = hparam
        self.pos_mat = self.get_pos_value()

    def get_pos_value(self):
        input_squese_shpae = hparams.max_length
        pos = tf.constant([i for i in range(input_squese_shpae)], dtype=tf.float32)
        pos = tf.reshape(pos, [input_squese_shpae, -1])
        local_mat = tf.cast(tf.range(self.hparam.d_model), dtype=tf.float32)[:, tf.newaxis]
        local_mat = 1 / tf.pow(10000, 2*local_mat / self.hparam.d_model)
        pos_mat = tf.matmul(pos, local_mat, transpose_b=True)
        pos_mat = pos_mat.numpy()
        pos_mat[:, ::2] = tf.sin(pos_mat[:, ::2])
        pos_mat[:, 1::2] = tf.cos(pos_mat[:, ::2])
        pos_mat = tf.convert_to_tensor(pos_mat, dtype=tf.float32)
        pos_mat = pos_mat[tf.newaxis, ...]
        return pos_mat

    def call(self, inputs, **kwargs):
        seq_length = inputs.shape[1]
        postion_value = self.pos_mat[:, :seq_length, :]
        return postion_value + inputs


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, hparam):
        super(MultiHeadAttention, self).__init__()
        self.hparam = hparam
        self.q = tf.keras.layers.Dense(hparam.head_num * hparam.head_depth)
        self.k = tf.keras.layers.Dense(hparam.head_num * hparam.head_depth)
        self.v = tf.keras.layers.Dense(hparam.head_num * hparam.head_depth)
        self.dense = tf.keras.layers.Layer(self.hparam.head_depth*self.hparam.head_num)

    def split_multi_head(self, mat):
        mat = tf.reshape(mat, [mat.shape[0], -1, self.hparam.head_num, self.hparam.head_depth])
        mat = tf.transpose(mat, [0, 2, 1, 3])
        return mat

    def call(self, inputs, **kwargs):
        input_q, input_k, input_v, mask = inputs[0], inputs[1], inputs[2], inputs[3]
        q = self.q(input_q)
        k = self.k(input_k)
        v = self.v(input_v)
        # 把多个head分开
        q = self.split_multi_head(q)
        k = self.split_multi_head(k)
        v = self.split_multi_head(v)

        qk = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.hparam.head_depth, dtype=tf.float32))

        qk += (mask * -1e9)

        value_weights = tf.nn.softmax(qk, axis=0)

        value = tf.matmul(value_weights, v)
        value = tf.transpose(value, [0, 2, 1, 3])
        value = tf.reshape(value, [inputs[1].shape[0], -1, self.hparam.head_depth*self.hparam.head_num])
        value = self.dense(value)
        return value


def mask_fun(inputs):
    mask = tf.cast(tf.equal(inputs, 0), dtype=tf.float32)
    mask = mask[:, tf.newaxis, tf.newaxis, :]
    return mask


def look_ahead_mask(inputs):
    seq_len = inputs.shape[1]
    # 变成去点主对角线的上三角函数
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)
    pad_mask = mask_fun(inputs)
    look_ahead_mask = tf.maximum(look_ahead_mask, pad_mask)
    return look_ahead_mask


class Encoder_Layer(tf.keras.layers.Layer):

    def __init__(self, hparam):
        super(Encoder_Layer, self).__init__()
        self.hparam = hparam
        self.attention = MultiHeadAttention(hparam)
        self.drop_out1 = tf.keras.layers.Dropout(hparam.drop_rate)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.dense1 = tf.keras.layers.Dense(hparam.d_model)
        self.dense2 = tf.keras.layers.Dense(hparam.d_model)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.drop_out2 = tf.keras.layers.Dropout(hparam.drop_rate)

    def call(self, inputs, **kwargs):
        # 第一部分multi-head attention
        input_1 = inputs[0]
        attention = self.attention(inputs)
        attention = self.drop_out1(attention)
        attention += input_1
        attention = self.norm1(attention)
        out = self.dense1(attention)
        out = self.dense2(out)
        out = self.drop_out2(out)
        out += attention
        out = self.norm2(out)
        return out


class Encoder(tf.keras.layers.Layer):
    def __init__(self, hparam):
        super(Encoder, self).__init__()
        self.hparam = hparam
        # 第一层 embedding 层 + mask 层
        self.encoder_embedding = tf.keras.layers.Embedding(self.hparam.vocab_size + 1, self.hparam.d_model)
        self.mask = tf.keras.layers.Lambda(mask_fun)
        # 第二层 position 层
        self.position = Position(self.hparam)
        self.drop_out = tf.keras.layers.Dropout(hparam.drop_rate)

        # 第三层 两层 encoder_layer
        self.encoder_layer_1 = Encoder_Layer(hparam)
        self.encoder_layer_2 = Encoder_Layer(hparam)

    def call(self, inputs, **kwargs):
        out = self.encoder_embedding(inputs)
        mask = self.mask(inputs)
        position = self.position(out)
        position = self.drop_out(position)
        out = self.encoder_layer_1([position, position, position, mask])
        out = self.encoder_layer_2([out, out, out, mask])
        return out


class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self, hparam):
        super(Decoder_Layer, self).__init__()
        self.hparam = hparam

        self.masked_attention = MultiHeadAttention(hparam)
        self.drop_out1 = tf.keras.layers.Dropout(hparam.drop_rate)
        self.norm1 = tf.keras.layers.LayerNormalization()

        self.attention = MultiHeadAttention(hparam)
        self.drop_out2 = tf.keras.layers.Dropout(hparam.drop_rate)
        self.norm2 = tf.keras.layers.LayerNormalization()

        self.dense1 = tf.keras.layers.Dense(hparam.d_model)
        self.dense2 = tf.keras.layers.Dense(hparam.d_model)
        self.drop_out3 = tf.keras.layers.Dropout(hparam.drop_rate)
        self.norm3 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, **kwargs):
        dec_input, en_input, look_ahead_mask, mask = inputs[0], inputs[1], inputs[2], inputs[3]

        attention_input = [dec_input, dec_input, dec_input, look_ahead_mask]
        mask_attention = self.masked_attention(attention_input)
        mask_attention += dec_input
        mask_attention = self.drop_out1(mask_attention)
        mask_attention = self.norm1(mask_attention)

        in_attention_input = [mask_attention, en_input, en_input, mask]
        out = self.attention(in_attention_input)
        out += mask_attention
        out = self.drop_out2(out)
        out = self.norm2(out)

        out2 = self.dense1(out)
        out2 = self.dense2(out2)
        out += out2
        out = self.drop_out3(out)
        out = self.norm3(out)
        return out


class Decoder(tf.keras.layers.Layer):
    def __init__(self, hparam):
        super(Decoder, self).__init__()
        self.hparam = hparam
        # 第一层 embedding 层 + mask 层
        self.embedding = tf.keras.layers.Embedding(self.hparam.vocab_size + 1, self.hparam.d_model)
        self.mask = tf.keras.layers.Lambda(mask_fun)
        self.look_ahead_mask = tf.keras.layers.Lambda(look_ahead_mask)
        self.drop_out = tf.keras.layers.Dropout(self.hparam.drop_rate)

        # 第二层 position 层
        self.position = Position(self.hparam)

        # 第三层 两层 encoder_layer
        self.decoder_layer_1 = Decoder_Layer(hparam)
        self.decoder_layer_2 = Decoder_Layer(hparam)

        self.linear = tf.keras.layers.Dense(self.hparam.d_model)
        self.linear_soft_max = tf.keras.layers.Dense(self.hparam.vocab_size)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, **kwargs):
        dec_input, en_input = inputs[0], inputs[1]
        mask_input = inputs[2]
        out = self.embedding(dec_input)
        mask = self.mask(mask_input)
        look_ahead_mask = self.look_ahead_mask(dec_input)
        position = self.position(out)
        position = self.drop_out(position)

        input1 = [position, en_input, look_ahead_mask, mask]
        out = self.decoder_layer_1(input1)
        input2 = [out, en_input, look_ahead_mask, mask]
        out = self.decoder_layer_2(input2)
        out = self.linear(out)
        out = self.linear_soft_max(out)
        out = self.softmax(out)
        return out


# 构建transformer
class Transformer(tf.keras.Model):

    def __init__(self, hparam):
        super(Transformer, self).__init__()
        self.hparam = hparam
        self.encoder = Encoder(hparam)
        self.decoder = Decoder(hparam)

    def call(self, inputs, training=None, mask=None):
        input_1, input_2 = inputs["inputs"], inputs["dec_inputs"]
        out_1 = self.encoder(input_1)
        out2 = self.decoder([input_2, out_1, input_1])
        return out2


if __name__ == "__main__":
    hparam = {}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_samples',
        default=2500,
        type=int,
        help='maximum number of conversation pairs to use')

    parser.add_argument(
        '--max_length', default=40, type=int, help='maximum sentence length')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_units', default=512, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--head_num', default=8, type=int)
    parser.add_argument('--drop_rate', default=0.1, type=float)
    parser.add_argument('--head_depth', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    hparams = parser.parse_args()

    dataset, tokenizer = get_dataset(hparams)
    model = Transformer(hparams)
    optimizer = tf.keras.optimizers.Adam(
        CustomSchedule(hparams), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer, loss=loss_function, metrics=[accuracy])
    data_one_step = dataset.as_numpy_iterator().next()
    X = data_one_step[0]
    X['inputs'] = tf.constant(X['inputs'], dtype=tf.float32)
    X['dec_inputs'] = tf.constant(X['dec_inputs'], dtype=tf.float32)
    y = data_one_step[1]
    y = tf.constant(y, dtype=tf.float32)
    model(X)
    model.train_on_batch(X, y)
    print('aaaa')