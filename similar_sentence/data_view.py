import tensorflow_hub as hub
import sentencepiece as spm
import numpy as np
import os
import scipy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
import tensorflow as tf


# data = pd.read_csv('atec_nlp_sim_train_add.csv', header=None, delimiter="\n")
def load_dataset(filename):
    sent_pairs = []
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            # print(ts[1], ts[2], ts[3])
            sent_pairs.append((ts[1], ts[2], float(ts[3])))
    return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


data = load_dataset('../atec_nlp_sim_train_add.csv')
data_train = data.iloc[:100]
data_test = data.iloc[60:76]
print(data_test)

data = data.sort_values(by='sim', ascending=False)
data_train = data.iloc[:100]
data_test = data.iloc[60:76]
print(data_test)

print('Start downlaod...')
module = hub.Module("/home/alex/my_module_cache/9c61abbea1e6365bdd67e17707f5dd2434ea42d7/")
# module = hub.Module("https://tfhub.dev/google/nnlm-zh-dim128-with-normalization/1")
print('End download...')

input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))
with tf.Session() as sess:
    spm_path = sess.run(module(signature="spm_path"))
sp = spm.SentencePieceProcessor()
sp.Load(spm_path)
print("SentencePiece model loaded at {}.".format(spm_path))


def process_to_IDs_in_sparse_format(sp, sentences):
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values = [item for sublist in ids for item in sublist]
    indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return values, indices, dense_shape


values1, indices1, dense_shape1 = process_to_IDs_in_sparse_format(sp, data_train['sent_1'].tolist())
values2, indices2, dense_shape2 = process_to_IDs_in_sparse_format(sp, data_train['sent_2'].tolist())
similarity_scores = data_train['sim'].tolist()


values3, indices3, dense_shape3 = process_to_IDs_in_sparse_format(sp, data_test['sent_1'].tolist())
values4, indices4, dense_shape4 = process_to_IDs_in_sparse_format(sp, data_test['sent_2'].tolist())
similarity_scores_test = data_test['sim'].tolist()
print('ground truth:')
print(similarity_scores_test)
# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings1 = session.run(
        encodings,
        feed_dict={
            input_placeholder.values: values1,
            input_placeholder.indices: indices1,
            input_placeholder.dense_shape: dense_shape1
        }
    )
    # input_placeholder.values: values2,
    # input_placeholder.indices: indices2,
    # input_placeholder.dense_shape: dense_shape2})
    message_embeddings2 = session.run(
        encodings,
        feed_dict={
            input_placeholder.values: values2,
            input_placeholder.indices: indices2,
            input_placeholder.dense_shape: dense_shape2
        }
    )

    test1 = session.run(
        encodings,
        feed_dict={
            input_placeholder.values: values3,
            input_placeholder.indices: indices3,
            input_placeholder.dense_shape: dense_shape3})

    test2 = session.run(
        encodings,
        feed_dict={
            input_placeholder.values: values4,
            input_placeholder.indices: indices4,
            input_placeholder.dense_shape: dense_shape4})


import matplotlib.pyplot as plt
plt.figure()
for i in range(len(test1)):
    plt.subplot(4, 4, i + 1)
    plt.scatter(test1[i], test2[i])
plt.show()


