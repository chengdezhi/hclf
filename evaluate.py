import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from load_data import prediction_with_threshold 
from sklearn.metrics import precision_recall_fscore_support

class Evaluation(object):
  def __init__(self, config, preds, labels):
    self.summaries = []
    preds = preds[:,2:-1]
    assert  len(preds[0,:]) == len(labels[0,:])
    #print(preds[0,:], labels[0,:], len(preds[0,:]), len(labels[0,:]))
    self.get_metric(preds, labels, average='micro', about='all')
    self.get_metric(preds, labels, average='weighted', about='all')
    if config.eval_layers:
      self.get_metric(preds[:, config.layer1-2], labels[:, config.layer1-2], average='micro', about='layer_1')
      self.get_metric(preds[:, config.layer1-2], labels[:, config.layer1-2], average='weighted', about='layer_1')
      self.get_metric(preds[:, config.layer2-2], labels[:, config.layer2-2], average='micro', about='layer_2')
      self.get_metric(preds[:, config.layer2-2], labels[:, config.layer2-2], average='weighted', about='layer_2')
      self.get_metric(preds[:, config.layer3-2], labels[:, config.layer3-2], average='micro', about='layer_3')
      self.get_metric(preds[:, config.layer3-2], labels[:, config.layer3-2], average='weighted', about='layer_3')

    if config.eval_trees:
      self.get_metric(preds[:, config.tree1-2], labels[:, config.tree1-2], average='micro', about='tree_1')
      self.get_metric(preds[:, config.tree1-2], labels[:, config.tree1-2], average='weighted', about='tree_1')
      self.get_metric(preds[:, config.tree2-2], labels[:, config.tree2-2], average='micro', about='tree_2')
      self.get_metric(preds[:, config.tree2-2], labels[:, config.tree2-2], average='weighted', about='tree_2')
      self.get_metric(preds[:, config.tree3-2], labels[:, config.tree3-2], average='micro', about='tree_3')
      self.get_metric(preds[:, config.tree3-2], labels[:, config.tree3-2], average='weighted', about='tree_3')

  def get_metric(self, preds, labels, average=None, about="all", data_type="dev"):
    precisions, recalls, fscores, _ = precision_recall_fscore_support(labels, preds, average=average)
    if about=="all":
      print('%s average precision recall f1-score: %f %f %f' % (average, precisions, recalls, fscores))
    f1_summary = tf.Summary(value=[tf.Summary.Value(tag='{}:{}:{}/f1'.format(data_type, about, average), simple_value=fscores)])
    self.summaries.append(f1_summary)

class Evaluator(object):
  def __init__(self, config, model):
    self.config = config 
    self.model = model
    self.loss = model.loss
    self.preds = model.preds
    self.scores = model.scores
    self.mlb = MultiLabelBinarizer()
    self.mlb.fit([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])

  def get_metric(self, preds, labels, average=None, about="all", data_type="dev"):
    precisions, recalls, fscores, _ = precision_recall_fscore_support(labels, preds, average=average)
    if about=="all":
      print('%s average precision recall f1-score: %f %f %f' % (average, precisions, recalls, fscores))
    f1_summary = tf.Summary(value=[tf.Summary.Value(tag='{}:{}:{}/f1'.format(data_type, about, average), simple_value=fscores)])
    self.summaries.append(f1_summary)

  def get_evaluation(self, sess, batch):
    batch_idx, batch_ds = batch
    feed_dict = self.model.get_feed_dict(batch, False)
    #for key, val in feed_dict.items():
    #  print(key, np.array(val).shape) 
    # print("feed_dict:", feed_dict)
    preds, scores = sess.run([self.preds, self.scores], feed_dict=feed_dict)
    preds = prediction_with_threshold(preds, scores, threshold=self.config.multilabel_threshold)
    preds = self.mlb.transform(preds)
    labels = batch_ds.data["y_seqs"] 
    return preds, labels
    
  def get_evaluation_from_batches(self, sess, batches):
    config = self.config
    elist = [self.get_evaluation(sess, batch) for batch in batches]
    preds = [elem[0] for elem in elist]
    labels = [elem[1] for elem in elist]
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    # print("preds, labels:", preds[0,:], labels[0,:], len(preds[0,:]), len(labels[0,:]))
    return Evaluation(config, preds, labels)
