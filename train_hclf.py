from __future__ import print_function
from nltk.corpus import reuters 
from collections import Counter
from load_data import prepare_data, get_word2idx, get_word2vec, prediction_with_threshold 

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.seq2seq import BasicDecoder, sequence_loss, GreedyEmbeddingHelper, dynamic_decode, TrainingHelper, \
    ScheduledEmbeddingTrainingHelper, tile_batch, BeamSearchDecoder, BahdanauAttention, AttentionWrapper
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
import pickle, json, os

os.environ["CUDA_VISIBLE_DEVICES"]="7" 
mlb = MultiLabelBinarizer()

tree_1 = np.array([2,3,4,5,6,7,8])
tree_2 = np.array([9,10,11,12,13,14,15])
tree_3 = np.array([16,17,18,19,20,21,22,23,24])

layer1 = np.array([2, 9, 16])
layer2 = np.array([3, 4, 10, 11, 17, 21])
layer3 = np.array([5, 6, 7, 8, 12, 13, 14, 15, 18, 19, 20, 22, 23, 24])
mlb.fit([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]])

# Parameters
learning_rate = 1e-3
training_iters = 2000
train_epoch = 100
display_step = 30
embedding_size = 300
beam_width = 5
multilabel_threshold = -3.0 
print("multilabel_threshold:", multilabel_threshold)
EOS = 25
PAD = 0
GO = 1    # root ? 
max_seq_length = 4

# Network Parameters
hidden_size = 300  # hidden layer num of features
n_classes = 26  # 23+3

# word_embedding/label_embedding 
word2idx = Counter(json.load(open("data/word2idx.json", "r"))["word2idx"])
vocab_size = len(word2idx)
word2vec = get_word2vec(word2idx)
idx2vec = {word2idx[word]: vec for word, vec in word2vec.items() if word in word2idx}
unk_embedding = np.random.multivariate_normal(np.zeros(embedding_size), np.eye(embedding_size))
emb_mat = np.array([idx2vec[idx] if idx in idx2vec else unk_embedding for idx in range(vocab_size)])
print("emb_mat:", emb_mat.shape)

word_embeddings = tf.constant(emb_mat, dtype=tf.float32)
label_embeddings = tf.get_variable(name="embeddings", shape=[n_classes, embedding_size], dtype=tf.float32)
x_input, x_mask_input, x_len, y_seqs, y_decode, y_len, y_mask_input, train_size  = prepare_data(data_type="train", word2idx=word2idx, test_true_label=True)   
t_x_input, t_x_mask_input, t_x_len, t_label_seq, t_y_decode, t_y_len, t_y_mask_input, test_size = prepare_data(data_type="test", word2idx=word2idx, test_true_label=True)   
# train_batch_size, test_batch_size = train_size, test_size
train_batch_size, test_batch_size = 60, 60
train_dataset = tf.data.Dataset.from_tensor_slices((x_input, x_mask_input, x_len, y_seqs, y_decode, y_len, y_mask_input))
train_dataset = train_dataset.shuffle(buffer_size=1000).repeat(train_epoch).batch(60)
test_dataset = tf.data.Dataset.from_tensor_slices((t_x_input, t_x_mask_input, t_x_len, t_label_seq, t_y_decode, t_y_len, t_y_mask_input))    
test_dataset = test_dataset.batch(60)

# check 
train_iter = train_dataset.make_one_shot_iterator().get_next()
test_iter = test_dataset.make_one_shot_iterator().get_next()

'''
# check dataset
cnt = 0
with tf.Session() as sess:
    try:
        while True:
            train_data = sess.run(train_iter)
            if cnt<=1: 
                print(train_data[0].shape)
                print(len(train_data))
            cnt += 1
    except tf.errors.OutOfRangeError:
        print("end!")

print("cnt:", cnt)
'''

x = tf.placeholder(tf.int32, [None, None])      # [batch_size, max_doc_len]
x_mask = tf.placeholder(tf.int32, [None, None])      # [batch_size, max_doc_len]
y = tf.placeholder(tf.int32, [None, max_seq_length])
y_mask = tf.placeholder(tf.int32, [None, max_seq_length])
y_decoder = tf.placeholder(tf.int32, [None, max_seq_length])
x_seq_length = tf.placeholder(tf.int32, [None])
y_seq_length = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32)

xx = tf.nn.embedding_lookup(word_embeddings, x)  # [None, DL, d]    
yy = tf.nn.embedding_lookup(label_embeddings, y_decoder) # [None, seq_l, d]    

# encode here
''' 
lstm = rnn.LayerNormBasicLSTMCell(hidden_size/2, dropout_keep_prob=keep_prob)
outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm, lstm, xx, dtype='float', sequence_length=x_seq_length)   
xx_context = tf.concat(outputs, 2)   # [None, DL, 2*hd]
xx_final = tf.concat(output_states, 1)  # [None, 2*hd]
x_mask = tf.cast(x_mask, "float")
first_attention = tf.reduce_mean(xx_context,  1)    # [None, 2*hd]
'''

lstm = rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob=keep_prob)
outputs, output_states = tf.nn.dynamic_rnn(lstm, xx, dtype='float', sequence_length=x_seq_length)   
xx_context =  outputs  # tf.concat(outputs, 2)   # [None, DL, 2*hd]
xx_final =  output_states[0]  # tf.concat(output_states, 1)  # [None, 2*hd]
x_mask = tf.cast(x_mask, "float")
first_attention = tf.reduce_mean(xx_context,  1)    # [None, 2*hd]
# decode 
output_l = layers_core.Dense(n_classes, use_bias=True)
encoder_state = rnn.LSTMStateTuple(xx_final, xx_final)
attention_mechanism = BahdanauAttention(hidden_size, memory=xx_context, memory_sequence_length=x_seq_length)

lstm = rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob=keep_prob)
cell = AttentionWrapper(lstm, attention_mechanism, output_attention=False)
cell_state = cell.zero_state(dtype=tf.float32, batch_size=train_batch_size)
cell_state = cell_state.clone(cell_state=encoder_state, attention=first_attention)
train_helper = TrainingHelper(yy, y_seq_length)
train_decoder = BasicDecoder(cell, train_helper, cell_state, output_layer=output_l)
decoder_outputs_train, decoder_state_train, decoder_seq_train = dynamic_decode(train_decoder, impute_finished=True)

# infer_decoder/beam_search  
tiled_inputs = tile_batch(xx_context, multiplier=beam_width)
tiled_sequence_length = tile_batch(x_seq_length, multiplier=beam_width)
tiled_first_attention = tile_batch(first_attention, multiplier=beam_width)
attention_mechanism = BahdanauAttention(hidden_size, memory=tiled_inputs, memory_sequence_length=tiled_sequence_length)
tiled_xx_final = tile_batch(xx_final, beam_width)
encoder_state2 = rnn.LSTMStateTuple(tiled_xx_final, tiled_xx_final)

# lstm = rnn.LayerNormBasicLSTMCell(hidden_sizei, dropout_keep_prob=keep_prob)
cell = AttentionWrapper(lstm, attention_mechanism, output_attention=False)
cell_state = cell.zero_state(dtype=tf.float32, batch_size=test_batch_size * beam_width)
cell_state = cell_state.clone(cell_state=encoder_state2, attention=tiled_first_attention)
infer_decoder = BeamSearchDecoder(cell, embedding=label_embeddings, start_tokens=[GO] * test_batch_size, end_token=EOS,
                                  initial_state=cell_state, beam_width=beam_width, output_layer=output_l)
decoder_outputs_infer, decoder_state_infer, decoder_seq_infer = dynamic_decode(infer_decoder, maximum_iterations=4)

# cost/evaluate/train
weights = tf.sequence_mask(y_seq_length, max_seq_length, dtype=tf.float32)
cost = sequence_loss(logits=decoder_outputs_train.rnn_output, targets=y, weights=weights)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# predicted_ids: [batch_size, sequence_length, beam_width]
pred = decoder_outputs_infer.predicted_ids
scores = decoder_state_infer.log_probs

saver = tf.train.Saver(max_to_keep=20)
save_step = 100 
save_path = "data/mlb_hclf"

init = tf.global_variables_initializer()

training_iters = train_size * train_epoch // train_batch_size
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step <= training_iters:
        x_input, x_mask_input, x_len, y_seqs, y_decode, y_len, y_mask_input = sess.run(train_iter)
        sess.run(optimizer, feed_dict={x: x_input, x_mask:x_mask_input, y: y_seqs, y_decoder: y_decode,  x_seq_length:x_len, keep_prob: 0.5,
                                       y_seq_length: y_len})
        if step % save_step == 0:
            saver.save(sess, save_path, global_step=step)
            print("sava model at ", save_path, step)
        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={x: x_input, x_mask:x_mask_input, y: y_seqs, y_decoder: y_decode,  x_seq_length:x_len,  keep_prob: 0.5,
                                       y_seq_length: y_len})
            print("Iter " + str(step) + ", batch Loss= " + str(loss))
            print('test')
            test_steps = test_size // test_batch_size + 1
            pad_steps = test_steps * test_batch_size - test_size 
            print("ts, pad_s:", test_steps, pad_steps)
            t_preds_c = [] 
            for ts in range(test_steps):
                start_id , end_id = ts*test_batch_size, (ts+1)*test_batch_size
                assert start_id < test_size 
                t_x_input_b, t_x_mask_input_b, t_x_len_b = t_x_input[start_id:end_id, :], t_x_mask_input[start_id:end_id,:], t_x_len[start_id:end_id]
                if ts == test_steps - 1: 
                    p_x_input_b, p_x_mask_input_b, p_x_len_b = t_x_input[:pad_steps, :], t_x_mask_input[:pad_steps,:], t_x_len[:pad_steps]
                    t_x_input_b, t_x_mask_input_b, t_x_len_b = np.concatenate([p_x_input_b, t_x_input_b],axis=0), np.concatenate([p_x_mask_input_b, t_x_mask_input_b], axis=0),\
                                                               np.concatenate([p_x_len_b, t_x_len_b], axis=0) 
                # print("checked", t_x_input_b.shape, type(t_x_len_b), t_x_mask_input_b.shape, t_x_len_b.shape)
                t_preds, t_scores = sess.run([pred, scores], feed_dict={x: t_x_input_b, x_mask:t_x_mask_input_b, x_seq_length:t_x_len_b, keep_prob: 1.0})
                print("t_scores:", t_scores[0,:])
                t_preds_trans = prediction_with_threshold(t_preds, t_scores, threshold=multilabel_threshold)
                # print("t_preds_trans:", t_preds_trans)
                t_preds_b = mlb.transform(t_preds_trans)
                # print(type(t_preds_b),  t_preds_b.shape)
                if ts == test_steps - 1:
                   t_preds_c.append(t_preds_b[:test_batch_size-pad_steps,:])
                else:
                   t_preds_c.append(t_preds_b)
            print("before concat")
            t_preds_b = np.concatenate(t_preds_c, axis=0)
            print("before concat")
            # t_label_seq = test_label_seq
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq, t_preds_b[:, 2:25], average='micro')
            print('micro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq, t_preds_b[:, 2:25], average='weighted')
            print('macro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer1 - 2], t_preds_b[:, layer1],
                                                                              average='micro')
            print('layer1 micro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer1 - 2], t_preds_b[:, layer1],
                                                                              average='weighted')
            print('layer1 macro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer2 - 2], t_preds_b[:, layer2],
                                                                              average='micro')
            print('layer2 micro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer2 - 2], t_preds_b[:, layer2],
                                                                              average='weighted')
            print('layer2 macro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer3 - 2], t_preds_b[:, layer3],
                                                                              average='micro')
            print('layer3 micro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer3 - 2], t_preds_b[:, layer3],
                                                                              average='weighted')
            print('layer3 macro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, tree_1 - 2], t_preds_b[:, tree_1],
                                                                              average='micro')
            print('tree_1 micro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, tree_1 - 2], t_preds_b[:, tree_1],
                                                                              average='weighted')
            print('tree_1 macro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, tree_2 - 2], t_preds_b[:, tree_2],
                                                                              average='micro')
            print('tree_2 micro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, tree_2 - 2], t_preds_b[:, tree_2],
                                                                              average='weighted')
            print('tree_2 macro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, tree_3 - 2], t_preds_b[:, tree_3],
                                                                              average='micro')
            print('tree_3 micro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, tree_3 - 2], t_preds_b[:, tree_3],
                                                                              average='weighted')
            print('tree_3 macro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
        step += 1
    saver.save(sess, save_path, global_step=step)
    print("Optimization Finished!")

