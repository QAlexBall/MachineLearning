import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import re
import seaborn as sns
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.cross_validation import train_test_split

def load_dataset(filename):
  sent_pairs = []
  with tf.gfile.GFile(filename, "r") as f:
    for line in f:
      ts = line.strip().split("\t")
      # print(ts[1], ts[2], ts[3])
      sent_pairs.append((ts[1], ts[2], float(ts[3])))
  return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])
data = load_dataset('../atec_nlp_sim_train.csv')
# train, test = train_test_split(data, test_size = 0.2)
data_test = data.iloc[:200]
print(data_test)

print('Start downlaod...')
module = hub.Module("/home/alex/my_module_cache/9c61abbea1e6365bdd67e17707f5dd2434ea42d7/")
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
  dense_shape=(len(ids), max_len)
  values=[item for sublist in ids for item in sublist]
  indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
  return (values, indices, dense_shape)

  
values3, indices3, dense_shape3 = process_to_IDs_in_sparse_format(sp, data_test['sent_1'].tolist())
values4, indices4, dense_shape4 = process_to_IDs_in_sparse_format(sp, data_test['sent_2'].tolist())
similarity_scores_test = data_test['sim'].tolist()
# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  test = session.run(
      encodings,
      feed_dict={input_placeholder.values: values3,
                input_placeholder.indices: indices3,
                input_placeholder.dense_shape: dense_shape3,
                input_placeholder.values: values4,
                input_placeholder.indices: indices4,
                input_placeholder.dense_shape: dense_shape4})
clf = joblib.load('./clf.pkl')
print(clf.predict(test))
