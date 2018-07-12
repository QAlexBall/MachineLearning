import os

import numpy as np
import sentencepiece as spm
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold

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
data_test = data.iloc[:100]
print(data_test)
data = data.sort_values(by='sim', ascending=False)
data_train = data.iloc[9000:12000]

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
print(similarity_scores)


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

    plt.scatter(test1, test2)

    plt.show()

    # print((message_embeddings1,message_embeddings2).shape)

    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(100, 100, 100), learning_rate='constant',
                        learning_rate_init=0.01, max_iter=10000, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                        solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
                        warm_start=False)
    traindata = []
    for i in range(0, len(message_embeddings1)):
        data = np.concatenate((message_embeddings1[i], message_embeddings2[i]))
        traindata.append(data)

    testdata = []
    for i in range(0, len(test1)):
        data = np.concatenate((test1[i], test2[i]))
        testdata.append(data)
    # print(message_embeddings1)
    # print(message_embeddings1[0])
    # print(message_embeddings1[0][0])
    # print(len(message_embeddings1))
    sel = VarianceThreshold(threshold=(.8 * (1 - .2)))
    traindata = sel.fit_transform(traindata)
    clf.fit(traindata, similarity_scores)
    print(clf.loss_)
    print(clf.predict(testdata))
    # joblib.dump(clf, 'clf.pkl')
'''

    # lg = LogisticRegression()
    # lg.fit(message_embeddings, similarity_scores)
    # print(lg.predict(test))
    # joblib.dump(clf, './clf1.pkl')

    # for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    #   print("Message: {}".format(data_train['sent_1'][i]))
    #   print("Embedding size: {}".format(len(message_embedding)))
    #   message_embedding_snippet = ", ".join(
    #       (str(x) for x in message_embedding[:3]))
    #   print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
sts_input1 = tf.sparse_placeholder(tf.int64, shape=(None, None))
sts_input2 = tf.sparse_placeholder(tf.int64, shape=(None, None))

# For evaluation we use exactly normalized rather than
# approximately normalized.
sts_encode1 = tf.nn.l2_normalize(
    module(
        inputs=dict(values=sts_input1.values,
                    indices=sts_input1.indices,
                    dense_shape=sts_input1.dense_shape)),
    axis=1)
sts_encode2 = tf.nn.l2_normalize(
    module(
        inputs=dict(values=sts_input2.values,
                    indices=sts_input2.indices,
                    dense_shape=sts_input2.dense_shape)),
    axis=1)

sim_scores = -tf.acos(tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1))

def run_sts_benchmark(session):
  """Returns the similarity scores"""
  scores = session.run(
      sim_scores,
      feed_dict={
          sts_input1.values: values1,
          sts_input1.indices:  indices1,
          sts_input1.dense_shape:  dense_shape1,
          sts_input2.values:  values2,
          sts_input2.indices:  indices2,
          sts_input2.dense_shape:  dense_shape2,
      })
  return scores


with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  scores = run_sts_benchmark(session)
# print(len(scores))
# for i in range(len(scores)):
#     print(scores[i])
# print(scores)
print(similarity_scores)
pearson_correlation = scipy.stats.pearsonr(scores, similarity_scores)
print('Pearson correlation coefficient = {0}\np-value = {1}'.format(
    pearson_correlation[0], pearson_correlation[1]))
'''