
# coding: utf-8

# In[1]:


from __future__ import print_function
import os
import tensorflow as tf
import collections
import numpy as np
import math
import time


# In[2]:

###코퍼스에 따라 주석체크###
corpus_filenm = "text8.txt"
#corpus_filenm = "morphs_namu_small.txt"
# corpus_filenm = "/hdd/data/namu-wiki/morphs_namu_small.txtad"

train_mode = True
# train_mode = False

batch_size = 128            # Number of examples in a mini-batch
embedding_size = 128        # Dimension of the embedding vector.
skip_window = 2             # How many words to consider left and right.
num_neg_sampled = 64        # Number of negative examples to sample.
vocabulary_size = 50000     # Size of vocabulary

# We pick a random validation set to sample nearest neighbors.
valid_size = 16
valid_window = 1000
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

###코퍼스에 따라 주석체크###
num_steps = 100000 #text8
#num_steps = 1000000 #morphs_namu_small


# In[3]:

# Step 1. Read corpus and tokenize words based on space.
def read_data(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        return ' '.join(lines)

text = read_data(corpus_filenm)
text_words = text.split()
data_size = len(text_words)
print('Data size', data_size)


# In[4]:

# Step 2. Build vocabulary using most common top k words, replace word with ids in the corpus.
def build_dataset(words, n_words):
    """
    :param words: list of all words in corpus
    :param n_words: vocabulary size
    :return:
    """
    unique = collections.Counter(words)     # python dict - key(word): value(freq)
    orders = unique.most_common(n_words - 1)
    count = [('UNK', 0)]
    count.extend(orders)

    # check vocabulary coverage
    total_freq = 0
    for word, freq in orders:
        total_freq += freq
    print("word coverage: %.2f%%" % (100.0 * total_freq / data_size))  # word coverage

    # build word2id dictionary and id2word reverse dictionary.
    word2id = dict()
    for word, _ in count:
        word2id[word] = len(word2id)
    id2word = dict(zip(word2id.values(), word2id.keys()))

    # build training data by replacing all words with word ids.
    data = list()
    for word in words:
        # if the word is not in the dictionary, index will be 0. (i.e. 'UNK')
        index = word2id.get(word, 0)
        data.append(index)

    return data, count, word2id, id2word

vocab_counts = []
data, count, word2id, id2word = build_dataset(text_words, vocabulary_size)


del text_words

print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [id2word[i] for i in data[:10]])


# In[5]:

# Step 3. 학습에 사용할 mini-batch 생성
def generate_batch(batch_size, skip_window):
    
    global data_index
    
    num_targets = skip_window * 2
    
    assert batch_size % num_targets == 0
    assert num_targets <= 2 * skip_window
    
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 
    buffer = collections.deque(maxlen=span)##########buffer에 문제가 있음 deque error가 계속 뜬다..
    
    if data_index + span > len(data):
        data_index = 0
        
        
    #buffer.append(data[data_index: data_index + span])
    #data_index += span
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    
    
    for i in range(batch_size // num_targets):
        context_words = [w for w in range(span) if w != skip_window]

        for j, context_word in enumerate(context_words):
            batch[i * num_targets + j ] = buffer[skip_window]#buffer[0][skip_window]        # center words  e.g. [2, 2, 2, 2]   #ㄹㅇ..?
            labels[i * num_targets + j, 0] = buffer[context_word]#buffer[0][context_word]   # context words e.g. [0, 1, 3, 4]

        if data_index == len(data):
            # reset data index
            for word in data[:span]:
                buffer.append(word)
        else:
            # adding words to buffer
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# In[6]:

data_index = 0
batch, labels = generate_batch(batch_size=8, skip_window=skip_window)
for i in range(8):
    print(batch[i], id2word[batch[i]],
          '->', labels[i, 0], id2word[labels[i, 0]])


# In[ ]:




# In[7]:

# Step 4: Build a skip-gram tensorflow graph.
graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)         # word_ids of validation words

    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # Important - convert word ids to embedding vectors
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss - NCE: Noise Contrasive Estimation
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch
    # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    # http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_neg_sampled,
                       num_classes=vocabulary_size))

    # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        0.1,            # Base learning rate.
        global_step,    # Current index into the dataset.
        num_steps,      # Decay step.
        0.95,           # Decay rate.
        staircase=True)

    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

    # Compute the cosine similarity between mini-batch examples and all embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    # validation dataset의 유사 단어 찾기
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
    print('similarity : ',similarity)
    # Add variable initializer
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()


# In[ ]:




# In[8]:

# Step 5. Begin training.
start = time.time()
if train_mode:
    with tf.Session(graph=graph) as session:
        # we must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, skip_window)
            feed_dict = {train_inputs: batch_inputs,
                         train_labels: batch_labels}

            # we perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run())
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()  # dimension: (16, 50000)
                for i in range(valid_size):
                    valid_word = id2word[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors

                    # index 1로 시작하는 이유: query 단어는 제외함
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]

                    log_str = 'Nearest to %s:' % valid_word
                    sim_log_str = ''
                    for k in range(top_k):
                        close_word = id2word[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                        sim_log_str = '%s %.4f' % (sim_log_str, sim[i, nearest[k]])
                    print(log_str)
                    print(sim_log_str)

        final_embeddings = normalized_embeddings.eval()

        # Save the variables to disk.
        save_path = saver.save(session, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)

    print("training time used {:.1f}s".format(time.time() - start))


# In[9]:

find_examples = []


# In[10]:

# Step 5-1. query파일을 읽어 target word를 저장한다.

###코퍼스에 따라 주석체크###
with open("query_words_text8.txt") as f: #query_words_text8
#with open("query_words_namu.txt",encoding='utf-8-sig') as f: #query_words_namu
    for i in f.readlines(): 
        line = i.splitlines()[0]
        find_word = word2id[line]
        find_examples.append(find_word)
#print(find_examples)


# In[11]:

find_size = len(find_examples)
print(find_size)


# In[12]:

# Step 6. Restore checked tf checkpoint, perform query words similarity calculation

i = 0

with tf.Session(graph=graph) as session:
    # Restore variables from disk.
    saver.restore(session, "/tmp/model.ckpt")
    print("Model restored.")
    final_embeddings = normalized_embeddings.eval()

    # TODO: You can do word similarity calculation by running tensorflow (TF) operation here.
    # TODO: You must define a TF operation before running.
    # TODO: Hint: See how the 'similarity' TF operation works.
    
    # Compute the cosine similarity between target examples and all embeddings
    
    find_dataset = tf.constant(find_examples, dtype=tf.int32)
    find_embeddings = tf.nn.embedding_lookup(final_embeddings, find_dataset)
    similarity = tf.matmul(find_embeddings, final_embeddings, transpose_b=True)
    
    ###코퍼스에 따라 주석체크###    
    f = open("result_text8.txt", 'w') #result_text8
    #f = open("result_namu.txt", 'w') #result_namu
    
    sim = similarity.eval()  # dimension: (16, 50000)
    for i in range(find_size):
        valid_word = id2word[find_examples[i]]
        top_k = 8  # number of nearest neighbors
        # index 1로 시작하는 이유: query 단어는 제외함
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        
        log_str = 'Nearest to %s:' % valid_word
        f.write(valid_word + '\n')
        
        
        sim_log_str = ''
        for k in range(top_k):
            close_word = id2word[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
            sim_log_str = '%s %.4f' % (sim_log_str, sim[i, nearest[k]])
            output_str = '%.4f %s' % (sim[i, nearest[k]], close_word)
            f.write(output_str+'\n')
            
        print(log_str)
        print(sim_log_str)
        f.write('\n')
    
    f.close()



# In[ ]:




# In[13]:

# You can save word vectors into files like this.
embedding_filenm = "word2vec.txt"
print("Writing word vectors into file..")
with open(embedding_filenm, "w") as f:
    for word, vec in zip(word2id.keys(), final_embeddings):
        out = word + ' ' + ' '.join([str(v) for v in list(vec)]) + "\n"
        f.write(out)
    print("Done!")


# In[ ]:




# In[ ]:



