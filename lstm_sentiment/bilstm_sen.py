import tensorflow as tf
import os
import numpy as np
import tensorflow_datasets as tfds

train_path = './Chinese_conversation_sentiment/sentiment_XS_30k.txt'
test_path = './Chinese_conversation_sentiment/sentiment_XS_test.txt'


def load_data(path):
    with open(path, 'r', encoding='utf-8')as f:
        data = f.readlines()
    return data


def split_label(data: str):
    label_list = []
    data_list = []
    for line in data:
        line = line.rstrip()
        data_split = line.split(',')
        if data_split[0] == 'positive':
            label_list.append(1)
            data_list.append(data_split[1])
        elif data_split[0] == 'negative':
            label_list.append(0)
            data_list.append(data_split[1])
    return label_list, data_list


def split_word(data):
    corpus = []
    for line in data:
        line_corpus = [char for char in line if char != ' ']
        corpus.append(line_corpus)
    return corpus


def transform_corpus(data):
    corpus = []
    for line in data:
        line_start = ''
        for char in line:
            line_start += char
            line_start += ' '
        line_start = line_start.rstrip(' ')
        corpus.append(line_start)
    return corpus


def get_train_test(train_path, test_path):
    train_data = load_data(train_path)
    test_data = load_data(test_path)

    train_label, train_data = split_label(train_data)
    test_label, test_data = split_label(test_data)

    train_corpus = split_word(train_data)
    test_corpus = split_word(test_data)

    train_corpus = transform_corpus(train_corpus)
    test_corpus = transform_corpus(test_corpus)

    return train_corpus, test_corpus, train_label, test_label


def data_token(tokenizer, train_corpus):
    train_token = []
    for train_line in train_corpus:
        train_sentence = [tokenizer.vocab_size + 1] + tokenizer.encode(train_line) + [tokenizer.vocab_size + 2]

        train_token.append(train_sentence)
    train_token = tf.keras.preprocessing.sequence.pad_sequences(
        train_token, maxlen=25, padding='post')
    return train_token


class BiLSTMSentiment(tf.keras.Model):

    def __init__(self, vocabulary_size):
        super(BiLSTMSentiment, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocabulary_size,
                                                   output_dim=256)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True),
                                                    input_shape=(25, 256), merge_mode='concat')
        self.drop_out = tf.keras.layers.Dropout(0.5)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.dense_out = tf.keras.layers.Dense(2)
        self.sort_out = tf.keras.layers.Softmax()

    def call(self, inputs):
        out1 = self.embedding(inputs)
        out2 = self.bilstm(out1)
        out3 = self.drop_out(out2)
        out4 = self.flatten(out3)
        out5 = self.dense(out4)
        out6 = self.dense_out(out5)
        out7 = self.sort_out(out6)
        return out7


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, hparams, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(hparams, dtype=tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * self.warmup_steps ** -1.5
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def trans_label(train_label):
    train_label_0 = np.array(train_label).reshape((-1, 1))
    train_label_1 = 1 - train_label_0
    train_label = np.hstack([train_label_0, train_label_1])
    return train_label


if __name__ == '__main__':
    train_corpus, test_corpus, train_label, test_label = get_train_test(train_path, test_path)
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        train_corpus + test_corpus, target_vocab_size=2 ** 13)
    train_token = data_token(tokenizer, train_corpus)
    test_token = data_token(tokenizer, test_corpus)
    train_label = trans_label(train_label)
    dataset = tf.data.Dataset.from_tensor_slices((train_token, train_label))
    dataset = dataset.batch(64)
    dataset = dataset.shuffle(len(train_corpus))
    vocabulary_size = tokenizer.vocab_size + 3
    model = BiLSTMSentiment(vocabulary_size)

    data_one_step = dataset.as_numpy_iterator().next()
    X = data_one_step[0]
    y = data_one_step[1]
    model(X)

    optimizer = tf.keras.optimizers.Adam(
        CustomSchedule(256), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer, loss=tf.keras.losses.binary_crossentropy, metrics=[tf.keras.metrics.Accuracy()])
    model.fit(dataset, epochs=100)

    print('aaa')
