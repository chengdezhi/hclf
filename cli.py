import os
import numpy as np
import tensorflow as tf
from main import main as m

flags = tf.app.flags
# device 
flags.DEFINE_string("gpu_ids", "7", "Run ID [0]")
flags.DEFINE_string("device_type", "gpu", "Run ID [0]")
flags.DEFINE_integer("gpu_idx", 0, "")

# training
flags.DEFINE_float("learning_rate", 1e-3, "learning_rate")
flags.DEFINE_float("keep_prob", 0.8, "keep_prob")
flags.DEFINE_integer("num_batches", 600, "")
flags.DEFINE_integer("batch_size", 60, "")
flags.DEFINE_integer("test_batch_size", 67, "")
# TODO check epoch 
flags.DEFINE_integer("num_epochs", 2, "")
flags.DEFINE_integer("log_period", 30, "")
flags.DEFINE_integer("eval_period", 30, "")
flags.DEFINE_integer("save_period", 30, "")
flags.DEFINE_integer("val_num_batches", 0, "")

# network 
flags.DEFINE_integer("word_embedding_size", 300, "")
flags.DEFINE_integer("label_embedding_size", 300, "")
flags.DEFINE_integer("hidden_size", 150, "")
flags.DEFINE_integer("beam_width", 5, "")
flags.DEFINE_float("multilabel_threshold", -3.0, "")
flags.DEFINE_integer("EOS", 25, "")
flags.DEFINE_integer("PAD", 0, "")
flags.DEFINE_integer("GO", 1, "")

# graph control
flags.DEFINE_string("mode", "train", "")
flags.DEFINE_string("model_name", "hclf_baseline", "")   # RCNN 
flags.DEFINE_integer("max_to_keep", 30, "")

flags.DEFINE_boolean("load", False, "load saved data? [True]")
flags.DEFINE_boolean("eval", True, "eval data? [True]")
flags.DEFINE_boolean("eval_trees", True, "eval trees? [True]")
flags.DEFINE_boolean("eval_layers", True, "eval layers? [True]")
flags.DEFINE_boolean("cluster", True, "eval data? [True]")
flags.DEFINE_boolean("debug", False, "debug")

# define hierarchical 
flags.DEFINE_integer("max_seq_length", 4, "")
flags.DEFINE_integer("n_classes", 26, "")

# dir
flags.DEFINE_string("out_dir", "out", "")
# flags.DEFINE_string("save_dir", "out/save", "")
# flags.DEFINE_string("log_dir", "out/log", "")

def main(_):
  config = flags.FLAGS
  if config.debug:
    config.num_batches = 2
    config.log_period = 1
    config.save_period = 1
    config.eval_period = 1
    config.batch_size = 3
    config.val_num_batches = 6
    config.out_dir = "check"
  print(config.test_batch_size)
  config.tree1 = np.array([2,3,4,5,6,7,8])
  config.tree2 = np.array([9,10,11,12,13,14,15])
  config.tree3 = np.array([16,17,18,19,20,21,22,23,24])

  config.layer1 = np.array([2, 9, 16])
  config.layer2 = np.array([3, 4, 10, 11, 17, 21])
  config.layer3 = np.array([5, 6, 7, 8, 12, 13, 14, 15, 18, 19, 20, 22, 23, 24])

  config.save_dir = os.path.join(config.out_dir, "save")
  config.log_dir = os.path.join(config.out_dir, "log")
  if not os.path.exists(config.out_dir):
    os.makedirs(config.out_dir)
  if not os.path.exists(config.save_dir):
    os.mkdir(config.save_dir)
  if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)
  
  os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_ids
  m(config)

if __name__=="__main__":
  tf.app.run()
