import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.seq2seq import BasicDecoder, sequence_loss, GreedyEmbeddingHelper, dynamic_decode, TrainingHelper, \
    ScheduledEmbeddingTrainingHelper, tile_batch, BeamSearchDecoder, BahdanauAttention, AttentionWrapper
from load_data import DataSet 

def get_model(config):
  with tf.name_scope("model_{}".format(config.model_name)) as scope, tf.device("/{}:{}".format(config.device_type, config.gpu_idx)):
    model = Model(config, scope)
  return model

class Model(object):
  def __init__(self, config, scope):
    self.scope = scope
    self.config = config
    max_seq_length = config.max_seq_length 
    self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                   initializer=tf.constant_initializer(0), trainable=False)
    self.x = tf.placeholder(tf.int32, [None, config.max_docs_length], name="x")      # [batch_size, max_doc_len]
    self.x_mask = tf.placeholder(tf.int32, [None, config.max_docs_length], name="x_mask")      # [batch_size, max_doc_len]
    self.y = tf.placeholder(tf.int32, [None, max_seq_length], name="y")
    self.y_mask = tf.placeholder(tf.int32, [None, max_seq_length], name="y_mask")
    self.y_decoder = tf.placeholder(tf.int32, [None, max_seq_length], name="y-decoder")
    self.x_seq_length = tf.placeholder(tf.int32, [None], name="x_seq_length")
    self.y_seq_length = tf.placeholder(tf.int32, [None], name="y_seq_length")
    self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    self.output_l = layers_core.Dense(config.n_classes, use_bias=True)
    if config.model_name == "hclf_baseline": config.decode_size = config.hidden_size
    else: config.decode_size = 2*config.hidden_size  
    self.lstm = rnn.LayerNormBasicLSTMCell(config.decode_size, dropout_keep_prob=config.keep_prob)  # lstm for decode 
    self.encode_lstm = rnn.LayerNormBasicLSTMCell(config.hidden_size, dropout_keep_prob=config.keep_prob) # lstm for encode 
    # TODO config.emb_mat 
    self.word_embeddings = tf.constant(config.emb_mat, dtype=tf.float32, name="word_embeddings")
    self.label_embeddings = tf.get_variable(name="label_embeddings", shape=[config.n_classes, config.label_embedding_size], dtype=tf.float32)
    self.xx = tf.nn.embedding_lookup(self.word_embeddings, self.x)  # [None, DL, d]    
    self.yy = tf.nn.embedding_lookup(self.label_embeddings, self.y_decoder) # [None, seq_l, d]    
    self._build_encode(config)
    self._build_train(config)
    self._build_infer(config)
    self._build_loss(config)
    #self.infer_set = set()
    self.summary = tf.summary.merge_all()
    self.summary = tf.summary.merge(tf.get_collection("summaries", scope=self.scope))
  
  def _build_encode(self, config):
    if config.model_name == "hclf_baseline":
      outputs, output_states = tf.nn.dynamic_rnn(self.encode_lstm, self.xx, dtype='float', sequence_length=self.x_seq_length)  
      self.check = outputs  
      self.xx_context = outputs  # tf.concat(outputs, 2)   # [None, DL, 2*hd]
      self.xx_final = output_states[1]  # lstm cell output_states: [c,h]
      # TODO x_mask 
      x_mask = tf.cast(self.x_mask, "float")
      self.first_attention = tf.reduce_mean(self.xx_context,  1)    # [None, 2*hd]

    if config.model_name == "hclf_bilstm":
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.encode_lstm, self.encode_lstm, self.xx, dtype="float", sequence_length=self.x_seq_length)
      # self.check = output_states
      self.xx_context = tf.concat(outputs, 2)   # [None, DL, 2*hd]
      self.xx_final = tf.concat([output_states[0][1], output_states[1][1]], 1)  # [None, 2*hd]
      # TODO x_mask 
      x_mask = tf.cast(self.x_mask, "float")
      self.first_attention = tf.reduce_mean(self.xx_context,  1)    # [None, 2*hd]
      self.check = self.first_attention
      
    if config.model_name.startswith("RCNN"):
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.encode_lstm, self.encode_lstm, self.xx, dtype="float", sequence_length=self.x_seq_length)
      self.xx_context = tf.concat(outputs, 2)   # [None, DL, 2*hd]
      #self.xx_final = tf.concat([output_states[0][1], output_states[1][1]], 1)  # [None, 2*hd]
      self.xx_final = tf.layers.max_pooling1d(self.xx_context, config.max_docs_length, 1)
      self.xx_final = tf.squeeze(self.xx_final)
      self.xx_final = tf.reshape(self.xx_final, [-1, 2*config.hidden_size])
      print("check:", self.xx_final.get_shape())
      self.first_attention = tf.reduce_mean(self.xx_context,  1)    # [None, 2*hd]
      self.check = self.xx_final

  def _build_train(self, config):
    # decode
    if config.model_name == "RCNN-flat":
      self.logits = tf.contrib.layers.fully_connected(self.xx_final, config.n_classes)
    else:
      encoder_state = rnn.LSTMStateTuple(self.xx_final, self.xx_final)
      attention_mechanism = BahdanauAttention(config.decode_size, memory=self.xx_context, memory_sequence_length=self.x_seq_length)
      cell = AttentionWrapper(self.lstm, attention_mechanism, output_attention=False)
      cell_state = cell.zero_state(dtype=tf.float32, batch_size=config.batch_size)
      cell_state = cell_state.clone(cell_state=encoder_state, attention=self.first_attention)
      train_helper = TrainingHelper(self.yy, self.y_seq_length)
      train_decoder = BasicDecoder(cell, train_helper, cell_state, output_layer=self.output_l)
      self.decoder_outputs_train, decoder_state_train, decoder_seq_train = dynamic_decode(train_decoder, impute_finished=True)
      self.logits = self.decoder_outputs_train.rnn_output

  def _build_infer(self, config):
    # infer_decoder/beam_search  
    if config.model_name == "hclf_bilstm": decode_size = 2*config.hidden_size
    else: decode_size = config.hidden_size  
    tiled_inputs = tile_batch(self.xx_context, multiplier=config.beam_width)
    tiled_sequence_length = tile_batch(self.x_seq_length, multiplier=config.beam_width)
    tiled_first_attention = tile_batch(self.first_attention, multiplier=config.beam_width)
    attention_mechanism = BahdanauAttention(config.decode_size, memory=tiled_inputs, memory_sequence_length=tiled_sequence_length)
    tiled_xx_final = tile_batch(self.xx_final, config.beam_width)
    encoder_state2 = rnn.LSTMStateTuple(tiled_xx_final, tiled_xx_final)
    cell = AttentionWrapper(self.lstm, attention_mechanism, output_attention=False)
    cell_state = cell.zero_state(dtype=tf.float32, batch_size = config.dev_size * config.beam_width)
    cell_state = cell_state.clone(cell_state=encoder_state2, attention=tiled_first_attention)
    infer_decoder = BeamSearchDecoder(cell, embedding=self.label_embeddings, start_tokens=[config.GO]*config.dev_size, end_token=config.EOS,
                                  initial_state=cell_state, beam_width=config.beam_width, output_layer=self.output_l)
    decoder_outputs_infer, decoder_state_infer, decoder_seq_infer = dynamic_decode(infer_decoder, maximum_iterations=config.max_seq_length)
    self.preds = decoder_outputs_infer.predicted_ids
    self.scores = decoder_state_infer.log_probs

  def _build_loss(self, config):
    # cost/evaluate/train
    weights = tf.sequence_mask(self.y_seq_length, config.max_seq_length, dtype=tf.float32)
    if config.model_name == "RCNN_flat":
      self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels = self.y) # TODO self.y at multi-labels input 
    else:
      self.loss = sequence_loss(logits=self.decoder_outputs_train.rnn_output, targets=self.y, weights=weights)
    tf.summary.scalar(self.loss.op.name, self.loss) 
    # TODO process compute_gradients() and apply_gradients() separetely 
    self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss, global_step=self.global_step)
    # predicted_ids: [batch_size, sequence_length, beam_width]

  def get_feed_dict(self, batch, is_train):
    batch_idx, batch_ds = batch
    # TODO
    batch_ds = batch_ds.data
    feed_dict = {}
    feed_dict[self.x] = batch_ds["x"]
    feed_dict[self.x_mask] = batch_ds["x_mask"]
    feed_dict[self.x_seq_length] = batch_ds["x_len"]
    feed_dict[self.y_decoder] = batch_ds["decode_inps"]
    feed_dict[self.y_seq_length] = batch_ds["y_len"]
    if is_train:
      feed_dict[self.keep_prob] = self.config.keep_prob
      feed_dict[self.y] = batch_ds["y_seqs"]
    else:
      feed_dict[self.keep_prob] = 1
    return feed_dict
 
