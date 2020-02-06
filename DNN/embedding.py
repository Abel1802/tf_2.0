'''
    word embedding using tensorflow2.0
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np

# step-1,load datasets(keras.datasets.imdb :分装好的，电影评论)
imdb = keras.datasets.imdb
vocab_size = 10000
index_from = 3
# 加载前10000个词语，词表的index从 3 开始算
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size, index_from=index_from)
# look up the datasets imdb
print("look up the datasets imdb".center(100, '-'))
print('train_data[0]: {}'.format(train_data[0]))
print('train_labels[0]: {}'.format(train_labels[0]))
print('train_data.shape: {}'.format(train_data.shape))
print('train_labels.shape {}'.format(train_data.shape))


# step-2 建立词表索引
# word_index is a dic, eg "good":40080
word_index = imdb.get_word_index()
# 偏移 + 3
word_index = {k:(v+3) for k, v in word_index.items()}
# index_word = {v:k for k,v in word_index.items()}
# print(index_word[5])
# 前三个存入特殊字符
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<END>'] = 3
index_word = {value:key for key, value in word_index.items()}
# 将train_data, test_data decode 成文字
def decode_data(text_ids):
    return ' '.join(index_word.get(word_id, "<UNK>") for word_id in text_ids)
print('look for the true data of train_data[0]: {}'.center(100, '-'))
print(decode_data(train_data[0]))


# step-4 数据padding
# 长度低于500的补全，高于500的截断
max_length = 500
# 使用Keras.preprocessing.sequence.pad_sequence 补全
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, # list of list
    value = word_index['<PAD>'],
    padding = 'post', # post，pre 一个在前，一个在后
    maxlen = max_length
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, # list of list
    value = word_index['<PAD>'],
    padding = 'post', # post，pre 一个在前，一个在后
    maxlen = max_length
)
print('train_data[0] after padding: ----------------')
print(train_data[0])


# step-5 定义模型
# 每一个word embedding 成16维的向量
embedding_dim = 16
batch_size = 128
model = keras.models.Sequential([
    # 1,define a matrix [vocab_size , embedding_dim]
    # 2,对于每一个句子[1,2,3,4...], max_length * embedding_dim
    # 3, return batch_size * max_length * embedding_dim
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    # (合并)batch_size * max_length * embedding_dim  -> batch_size * embedding_dim
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
print('model.summary: ----------------')
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# step-6 train
print('train:'.center(100, '-'))
history = model.fit(train_data, train_labels,
                    epochs = 30,
                    batch_size = batch_size,
                    validation_split = 0.2)


# step-7 evaluate
print('evaluate:'.center(100, '-'))
model.evaluate(test_data, test_labels,
               batch_size = batch_size,)


