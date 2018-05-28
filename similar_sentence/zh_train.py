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
data_train = data.iloc[:20]
data_test = data.iloc[:10]
print(data_test)


print('Start downlaod...')
# module = hub.Module("/home/alex/my_module_cache/9c61abbea1e6365bdd67e17707f5dd2434ea42d7/")
module = hub.Module("https://tfhub.dev/google/nnlm-zh-dim128-with-normalization/1")
print('End download...')


