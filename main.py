import json, math
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf 
from collections import Counter
from pprint import pprint
from model import Model, get_model
from graph_handler import GraphHandler
from evaluate import Evaluator 
from load_data import read_data,get_word2vec

def main(config):
  with tf.device("/gpu:0"):
    if config.mode == "train":
      _train(config)
    elif config.mode == "test":
      _test(config)
    elif config.mode == "forward":
      _forward(config)
    else:
      raise ValueError("invalid value for 'mode': {}".format(config.mode))

def _train(config):
  word2idx = Counter(json.load(open("data/word2idx.json", "r"))["word2idx"])
  vocab_size = len(word2idx)
  word2vec = {} # or get_word2vec(word2idx)
  idx2vec = {word2idx[word]: vec for word, vec in word2vec.items() if word in word2idx}
  unk_embedding = np.random.multivariate_normal(np.zeros(config.word_embedding_size), np.eye(config.word_embedding_size))
  config.emb_mat = np.array([idx2vec[idx] if idx in idx2vec else unk_embedding for idx in range(vocab_size)])
  print("emb_mat:", config.emb_mat.shape)

  train_data = read_data(data_type="train", word2idx=word2idx, test_true_label=True)
  dev_data = read_data(data_type="test", word2idx=word2idx, test_true_label=True)
  
  pprint(config.__flags, indent=2)
  model = get_model(config)
  graph_handler = GraphHandler(config, model)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  graph_handler.initialize(sess)

  num_batches = config.num_batches or int(math.ceil(train_data.num_examples / config.batch_size)) * config.num_epochs
  global_step = 0

  dev_evaluate = Evaluator(config, model) 

  for batch in tqdm(train_data.get_batches(config.batch_size, num_batches=num_batches, shuffle=True, cluster=config.cluster), total=num_batches):
    batch_idx, batch_ds = batch
    global_step = sess.run(model.global_step) + 1
    # print("global_step:", global_step)
    get_summary = global_step % config.log_period  
    feed_dict = model.get_feed_dict(batch, config)
    loss, summary, train_op = sess.run([model.loss, model.summary, model.train_op], feed_dict=feed_dict)
    if get_summary:
      graph_handler.add_summary(summary, global_step)
    # occasional saving
    if global_step % config.save_period == 0:
      graph_handler.save(sess, global_step=global_step)
    if not config.eval:
      continue
    # Occasional evaluation
    if global_step % config.eval_period == 0:
      num_steps = math.ceil(dev_data.num_examples / config.test_batch_size)
      if 0 < config.val_num_batches < num_steps:
        num_steps = config.val_num_batches
      print("num_steps:", num_steps)
      e_dev = dev_evaluate.get_evaluation_from_batches(
        sess, tqdm(dev_data.get_batches(config.test_batch_size, num_batches=num_steps), total=num_steps))
      graph_handler.add_summaries(e_dev.summaries, global_step)
      
def _test(config):
  pass

def _forward(config):
  pass