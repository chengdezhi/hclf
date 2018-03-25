import gzip
import json
from json import encoder
import os
import tensorflow as tf

class GraphHandler(object):
  def __init__(self, config, model):
    self.config = config
    self.model = model
    self.saver = tf.train.Saver(max_to_keep=config.max_to_keep)
    self.writer = None
    self.save_path = os.path.join(config.save_dir, config.model_name)

  def initialize(self, sess):
    sess.run(tf.initialize_all_variables())
    print("initialize")
    if self.config.load:
      print("load sess")
      self._load(sess)

    if self.config.mode == 'train':
      self.writer =  tf.summary.FileWriter(self.config.log_dir, graph=tf.get_default_graph())

  def save(self, sess, global_step=None):
    saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
    saver.save(sess, self.save_path, global_step=global_step)

  def _load(self, sess):
    config = self.config
    vars_ = {var.name.split(":")[0]: var for var in tf.all_variables()}
    #for var in tf.all_variables():
    #  print("var:", var.name)
    #  break
    if config.load_ema:
      ema = self.model.var_ema
      for var in tf.trainable_variables():
        name = var.name.split(":")[0]
        if name in vars_:
          del vars_[name]
          vars_[ema.average_name(var)] = var

    saver = tf.train.Saver(vars_, max_to_keep=config.max_to_keep)

    if config.load_path:
      print (config.load_path)
      save_path = config.load_path
    elif config.load_step > 0:
      save_path = os.path.join(config.save_dir, "{}-{}".format(config.model_name, config.load_step))
    else:
      save_dir = config.save_dir
      checkpoint = tf.train.get_checkpoint_state(save_dir)
      assert checkpoint is not None, "cannot load checkpoint at {}".format(save_dir)
      save_path = checkpoint.model_checkpoint_path
    print("Loading saved model from {}".format(save_path))
    saver.restore(sess, save_path)

  def add_summary(self, summary, global_step):
    self.writer.add_summary(summary, global_step)

  def add_summaries(self, summaries, global_step):
    for summary in summaries:
      self.add_summary(summary, global_step)

