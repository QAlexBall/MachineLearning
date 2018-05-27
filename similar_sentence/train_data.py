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
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression


# data = pd.read_csv('atec_nlp_sim_train_add.csv', header=None, delimiter="\n")
def load_dataset(filename):
  sent_pairs = []
  with tf.gfile.GFile(filename, "r") as f:
    for line in f:
      ts = line.strip().split("\t")
      # print(ts[1], ts[2], ts[3])
      sent_pairs.append((ts[1], ts[2], float(ts[3])))
  return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])

data = load_dataset('atec_nlp_sim_train_add.csv')
data_train = data.iloc[:20000]
data_test = data.iloc[60001:60011]
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


values1, indices1, dense_shape1 = process_to_IDs_in_sparse_format(sp, data_train['sent_1'].tolist())
values2, indices2, dense_shape2 = process_to_IDs_in_sparse_format(sp, data_train['sent_2'].tolist())
similarity_scores = data_train['sim'].tolist()

values3, indices3, dense_shape3 = process_to_IDs_in_sparse_format(sp, data_test['sent_1'].tolist())
values4, indices4, dense_shape4 = process_to_IDs_in_sparse_format(sp, data_test['sent_2'].tolist())
similarity_scores_test = data_test['sim'].tolist()
print(similarity_scores_test)
# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(
      encodings,
      feed_dict={input_placeholder.values: values1,
                input_placeholder.indices: indices1,
                input_placeholder.dense_shape: dense_shape1,
                input_placeholder.values: values2,
                input_placeholder.indices: indices2,
                input_placeholder.dense_shape: dense_shape2})

  test = session.run(
      encodings,
      feed_dict={input_placeholder.values: values3,
                input_placeholder.indices: indices3,
                input_placeholder.dense_shape: dense_shape3,
                input_placeholder.values: values4,
                input_placeholder.indices: indices4,
                input_placeholder.dense_shape: dense_shape4})

  clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(50, 3), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
  clf.fit(message_embeddings, similarity_scores)
  print(clf.predict(test))


  lg = LogisticRegression()
  lg.fit(message_embeddings, similarity_scores)
  print(lg.predict(test))
  joblib.dump(clf, './clf.pkl')


  # for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
  #   print("Message: {}".format(data_train['sent_1'][i]))
  #   print("Embedding size: {}".format(len(message_embedding)))
  #   message_embedding_snippet = ", ".join(
  #       (str(x) for x in message_embedding[:3]))
  #   print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
