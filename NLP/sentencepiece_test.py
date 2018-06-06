import sentencepiece as spm


def process_to_IDs_in_sparse_format(sp, sentences):
    # An utility method that processes sentences with the sentence piece processor
    # 'sp' and returns the results in tf.SparseTensor-similar format:
    # (values, indices, dense_shape)
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values = [item for sublist in ids for item in sublist]
    indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)


sp = spm.SentencePieceProcessor()
# sp.Load("/path/to/sentence_piece/model")

module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/1")
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"]

input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
embeddings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))

values, indices, dense_shape = process_to_IDs_in_sparse_format(sentences)

message_embeddings = session.run(
    embeddings,
    feed_dict={input_placeholder.values: values,
               input_placeholder.indices: indices,
               input_placeholder.dense_shape: dense_shape})

print(message_embeddings)
