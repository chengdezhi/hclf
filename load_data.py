from nltk.corpus import stopwords, reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter
import numpy as np
from collections import Counter
import re, json, os, math, random, itertools
from itertools import zip_longest
from collections import Counter, defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

def grouper(iterable, n, fillvalue=None, shorten=False, num_groups=None):
  args = [iter(iterable)] * n
  out = zip_longest(*args, fillvalue=fillvalue)
  out = list(out)
  if num_groups is not None:
    default = (fillvalue, ) * n
    assert isinstance(num_groups, int)
    out = list(each for each, _ in zip_longest(out, range(num_groups), fillvalue=default))
  if shorten:
    assert fillvalue is None
    out = (tuple(e for e in each if e is not None) for each in out)
  return out

class DataSet(object):
  def __init__(self, data, data_type, valid_idxs= None):
    self.data = data  # {x:[], y:[]}  etc
    self.data_type = data_type
    total_num_examples = self.get_data_size()
    self.valid_idxs = range(total_num_examples) if valid_idxs is None else valid_idxs
    self.num_examples = len(self.valid_idxs)
   
  def _sort_key(self, idx):
    x = self.data["x"][idx]
    return len(x)

  def get_data_size(self):
    assert isinstance(self.data, dict)
    return len(next(iter(self.data.values())))

  def get_by_idxs(self, idxs):
    assert isinstance(self.data, dict)
    out = defaultdict(list)
    for key, val in self.data.items():
      out[key].extend(val[idx] for idx in idxs)
    return out

  def get_batches(self, batch_size, num_batches=None, shuffle=False, cluster=False):
    num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
    if num_batches is None:
      num_batches = num_batches_per_epoch
    num_epochs = int(math.ceil(num_batches / num_batches_per_epoch))
    
    if shuffle:
      random_idxs = random.sample(self.valid_idxs, len(self.valid_idxs)) # shuffle 
      if cluster:
        sorted_idxs = sorted(random_idxs, key=self._sort_key)
        sorted_grouped = lambda: list(grouper(sorted_idxs, batch_size))
        grouped = lambda: random.sample(sorted_grouped(), num_batches_per_epoch)
      else:
        random_grouped = lambda: list(grouper(random_idxs, batch_size))
        grouped = random_grouped
    else:
      raw_grouped = lambda: list(grouper(self.valid_idxs, batch_size))
      grouped = raw_grouped

    batch_idx_tuples = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))
    
    for _ in range(num_batches):
      batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
      batch_data = self.get_by_idxs(batch_idxs)
      batch_ds = DataSet(batch_data, self.data_type)
      print("batch_idxs:", len(batch_idxs))
      yield batch_idxs, batch_ds

def read_data(config, data_type="train", word2idx=None, max_seq_length=4):
  print("preparing {} data".format(data_type)) 
  docs, label_seqs, decode_inps, seq_lens = load_hclf_reuters(data_type=data_type)
  docs = [tokenize(reuters.raw(doc_id)) for doc_id in docs]
  docs_filter = [] 
  filter_ids = []
  for doc in docs:
    if len(doc)>0: 
      docs_filter.append(doc)
      filter_ids.append(1)
    else:
      filter_ids.append(0)
  docs = docs_filter
  docs_lens = [len(doc) for doc in docs]
  max_docs_length = 0
  
  for doc in docs:
    # print(len(doc))
    config.max_docs_length = len(doc) if len(doc) > config.max_docs_length else config.max_docs_length
  
  print("max_doc_length:", data_type, config.max_docs_length)
  docs2mat = [[word2idx[doc[_]] if _ < len(doc) else 1 for _ in range(config.max_docs_length)] for doc in docs] 
  docs2mask = [[1 if _ < len(doc) else 0 for _ in range(config.max_docs_length)] for doc in docs] 
  
  label_seqs_f, decode_inps_f, seq_lens_f = [], [], []  # for filter 
  for label_seq, decode_inp, seq_len, flag in zip(label_seqs, decode_inps, seq_lens, filter_ids):
    if flag==1:
      label_seqs_f.append(label_seq)
      decode_inps_f.append(decode_inp)
      seq_lens_f.append(seq_len)
  
  label_seqs, decode_inps, seq_lens = label_seqs_f, decode_inps_f, seq_lens_f 
  y_seq_mask = [[1 if i<sl else 0 for i in range(max_seq_length)] for sl in seq_lens]
  print(data_type, len(seq_lens))
  data = {
          "x": docs2mat, 
          "x_mask":docs2mask,
          "x_len": docs_lens,
          "y_seqs":label_seqs,
          "decode_inps": decode_inps,
          "y_mask": y_seq_mask,
          "y_len": seq_lens
         }
  return DataSet(data, data_type)

def tokenize(text):
  min_length = 2
  tokens = map(lambda word: word.lower(), word_tokenize(text))
  cachedStopWords = stopwords.words("english")
  tokens = [word for word in tokens if word not in cachedStopWords]
  tokens = (list(map(lambda token: PorterStemmer().stem(token), tokens)))
  p = re.compile('[a-zA-Z]+');
  filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length,tokens))
  return filtered_tokens


def get_word2vec(word_counter):
  glove_path = os.path.join("/home/t-dechen/data/glove", "glove.{}.{}d.txt".format("6B", 300))
  sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
  total = sizes['6B']
  word2vec_dict = {}
  with open(glove_path, 'r', encoding='utf-8') as fh:
      for line in tqdm(fh, total=total):
          array = line.lstrip().rstrip().split(" ")
          word = array[0]
          vector = list(map(float, array[1:]))
          if word in word_counter:
              word2vec_dict[word] = vector
          elif word.capitalize() in word_counter:
              word2vec_dict[word.capitalize()] = vector
          elif word.lower() in word_counter:
              word2vec_dict[word.lower()] = vector
          elif word.upper() in word_counter:
              word2vec_dict[word.upper()] = vector

  print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
  return word2vec_dict

def load_data():
  word2idx = Counter(json.load(open("data/word2idx.json", "r"))["word2idx"])
  prepare_data(data_type="train", word2idx=word2idx, test_true_label=False) 
  prepare_data(data_type="test", word2idx=word2idx, test_true_label=False) 


def get_word2idx():
  docs, label_seqs, decode_inp, seq_len = load_hclf_data(data_type="train")
  docs = [tokenize(reuters.raw(doc_id)) for doc_id in docs]
  max_docs_length = 0
  
  word2idx = Counter()
  word2idx["UNK"] = 0
  word2idx["NULL"] = 1  # for pad 
  for doc in docs:
      max_docs_length = len(doc) if len(doc) > max_docs_length else max_docs_length
      for token in doc:
          word2idx[token] = len(word2idx)
  
  shared = {"word2idx": word2idx}
  json.dump(shared, open("data/word2idx.json", "w"))
    
    
def prepare_data(data_type="train", word2idx=None, max_seq_length=4, test_true_label=False):
  print("preparing {} data".format(data_type), "test_true_label:", test_true_label) 
  docs, label_seqs, decode_inps, seq_lens = load_hclf_data(data_type=data_type, test_true_label=test_true_label)
  docs = [tokenize(reuters.raw(doc_id)) for doc_id in docs]
  docs_filter = [] 
  filter_ids = []
  for doc in docs:
    if len(doc)>0: 
      docs_filter.append(doc)
      filter_ids.append(1)
    else:
      filter_ids.append(0)
  docs = docs_filter
  docs_len = [len(doc) for doc in docs]
  max_docs_length = 0
  
  for doc in docs:
    max_docs_length = len(doc) if len(doc) > max_docs_length else max_docs_length

  docs2mat = [[word2idx[doc[_]] if _ < len(doc) else 1 for _ in range(max_docs_length)] for doc in docs] 
  docs2mask = [[1 if _ < len(doc) else 0 for _ in range(max_docs_length)] for doc in docs] 
  
  label_seqs_f, decode_inps_f, seq_lens_f = [], [], []  # for filter 
  for label_seq, decode_inp, seq_len, flag in zip(label_seqs, decode_inps, seq_lens, filter_ids):
    if flag==1:
      label_seqs_f.append(label_seq)
      decode_inps_f.append(decode_inp)
      seq_lens_f.append(seq_len)
  
  label_seqs, decode_inps, seq_lens = label_seqs_f, decode_inps_f, seq_lens_f 
  y_seq_mask = [[1 if i<sl else 0 for i in range(max_seq_length)] for sl in seq_lens]
  # print(docs2mat[0])
  # print(data_type, max_docs_length)
  print(data_type, len(seq_lens))
  return np.array(docs2mat), np.array(docs2mask), np.array(docs_len), np.array(label_seqs), np.array(decode_inps), np.array(seq_lens), np.array(y_seq_mask), len(seq_lens)
     
def load_hclf_reuters(data_type="train", allow_internal=True, in_hierarchy=True):
  layer_1 = ["hier1",  "hier2", "hier3"]
  layer_2 = ["grain", "crude", "livestock", "veg-oil", "meal-feed", "strategic-metal"]
  layer_3 = ["corn",  "wheat", "ship", "nat-gas", "carcass", "hog", "oilseed", "palm-oil", "barley", "rice", "cocoa", "copper", "tin", "iron-steel"]

  label2id = {
              "grain":3, "crude":4, "livestock":10, "veg-oil":11, "meal-feed":17, "strategic-metal":21, 
              "corn":5,  "wheat":6, "ship":7, "nat-gas":8, 
              "carcass":12, "hog":13, "oilseed":14, "palm-oil":15, 
              "barley":18, "rice":19, "cocoa":20, "copper":22, 
              "tin":23, "iron-steel":24
             }
  
  tree_labels = set(layer_2 + layer_3)
  sl_labels   = set(layer_2)
  leaf_labels = set(layer_3)

  # 23
  seqs    = [
             [2,3], [2,4], [9,10], [9,11], [16,17], [16,21],
             [2,3,5], [2,3,6], [2,4,7], [2,4,8], [9,10,12], [9,10,13], [9,11,14], 
             [9,11,15], [16,17,18], [16,17,19], [16,17,20], [16,21,22], [16,21,23], [16,21,24]
            ]

  targets = [
             [2,3,25,0], [2,4,25,0], [9,10,25,0], [9,11,25,0], [16,17,25,0], [16,21,25,0],
             [2,3,5,25], [2,3,6,25], [2,4,7,25], [2,4,8,25], [9,10,12,25], [9,10,13,25], [9,11,14,25], 
             [9,11,15,25], [16,17,18,25], [16,17,19,25], [16,17,20,25], [16,21,22,25], [16,21,23,25], [16,21,24,25]
            ]

  d_inputs = [
             [2,3,0], [2,4,0], [9,10,0], [9,11,0], [16,17,0], [16,21,0],
             [2,3,5], [2,3,6], [2,4,7], [2,4,8], [9,10,12], [9,10,13], [9,11,14], 
             [9,11,15], [16,17,18], [16,17,19], [16,17,20], [16,21,22], [16,21,23], [16,21,24]
            ]

  tree_1 = set([3,4,5,6,7,8])
  tree_2 = set([10,11,12,13,14,15])
  tree_3 = set([17,18,19,20,21,22,23,24])
   
  
  docs = []
  label_seqs = []
  decode_inp = []
  seq_len = []

  def _process(doc_id, seq, target, d_input):
      docs.append(doc_id)
      label_seqs.append(target)
      decode_inp.append([1] +d_input)
      seq_len.append(len(seq)+1)


  mlb = MultiLabelBinarizer()
  documents = reuters.fileids()
  documents = list(filter(lambda doc: doc.startswith(data_type),  documents))

  cnt = 0
  check_cnt = 0
  positive_cnt = 0
   
  for doc_id in documents:
      doc_labels = set([label2id[_] for _ in reuters.categories(doc_id) if _ in label2id])
      # print(doc_labels)
      doc_hcls = []
      hit = False 
      for seq, target, d_input in list(zip(seqs[::-1], targets[::-1], d_inputs[::-1])):
          keep = len(set(seq[1:]) - doc_labels) == 0
          # if positive_cnt < 10 : print(keep)
          if keep:   # hit 
            hit = True
            repeat = False
            for doc_hcl in doc_hcls: 
                if len(set(seq)-set(doc_hcl))==0: repeat=True
            if not repeat: 
                doc_hcls.append(seq)
                if data_type=="train": _process(doc_id, seq, target, d_input)
      if hit : positive_cnt += 1
            
      if data_type == "test" and hit:
          tl = set()
          for doc_hcl in doc_hcls: tl = tl | set(doc_hcl)
          tl = list(tl)
          label_seqs.append(tl)
          docs.append(doc_id)
          decode_inp.append([1,0,0,0])   # 1 start 
          seq_len.append(1)
  print(data_type, len(documents), len(docs), positive_cnt)
  if data_type=="test":  label_seqs = mlb.fit_transform(label_seqs)
  return docs, label_seqs, decode_inp, seq_len                 

def get_fasttext(f, data_type="train", allow_internal=True, in_hierarchy=True):
  layer_1 = ["hier1",  "hier2", "hier3"]
  layer_2 = ["grain", "crude", "livestock", "veg-oil", "meal-feed", "strategic-metal"]
  layer_3 = ["corn",  "wheat", "ship", "nat-gas", "carcass", "hog", "oilseed", "palm-oil", "barley", "rice", "cocoa", "copper", "tin", "iron-steel"]

  label2id = {
              "grain":3, "crude":4, "livestock":10, "veg-oil":11, "meal-feed":17, "strategic-metal":21, 
              "corn":5,  "wheat":6, "ship":7, "nat-gas":8, 
              "carcass":12, "hog":13, "oilseed":14, "palm-oil":15, 
              "barley":18, "rice":19, "cocoa":20, "copper":22, 
              "tin":23, "iron-steel":24
             }
  
  # 23
  seqs    = [
             [2,3], [2,4], [9,10], [9,11], [16,17], [16,21],
             [2,3,5], [2,3,6], [2,4,7], [2,4,8], [9,10,12], [9,10,13], [9,11,14], 
             [9,11,15], [16,17,18], [16,17,19], [16,17,20], [16,21,22], [16,21,23], [16,21,24]
            ]

  targets = [
             [2,3,25,0], [2,4,25,0], [9,10,25,0], [9,11,25,0], [16,17,25,0], [16,21,25,0],
             [2,3,5,25], [2,3,6,25], [2,4,7,25], [2,4,8,25], [9,10,12,25], [9,10,13,25], [9,11,14,25], 
             [9,11,15,25], [16,17,18,25], [16,17,19,25], [16,17,20,25], [16,21,22,25], [16,21,23,25], [16,21,24,25]
            ]

  d_inputs = [
             [2,3,0], [2,4,0], [9,10,0], [9,11,0], [16,17,0], [16,21,0],
             [2,3,5], [2,3,6], [2,4,7], [2,4,8], [9,10,12], [9,10,13], [9,11,14], 
             [9,11,15], [16,17,18], [16,17,19], [16,17,20], [16,21,22], [16,21,23], [16,21,24]
            ]

  mlb = MultiLabelBinarizer()
  documents = reuters.fileids()
  documents = list(filter(lambda doc: doc.startswith(data_type),  documents))
  cnt = 0
  check_cnt = 0
  positive_cnt = 0
   
  label2cnt = defaultdict(int)
  for doc_id in documents:
      doc_labels = set([label2id[_] for _ in reuters.categories(doc_id) if _ in label2id])
      # print(doc_labels)
      doc_hcls = []
      hit = False 
      for seq, target, d_input in list(zip(seqs[::-1], targets[::-1], d_inputs[::-1])):
          keep = len(set(seq[1:]) - doc_labels) == 0
          # if positive_cnt < 10 : print(keep)
          if keep:   # hit 
            hit = True
            repeat = False
            for doc_hcl in doc_hcls: 
                if len(set(seq)-set(doc_hcl))==0: repeat=True
            if not repeat: 
                doc_hcls.append(seq)
      if hit : positive_cnt += 1
            
      if hit:
          tl = set()
          for doc_hcl in doc_hcls: tl = tl | set(doc_hcl)
          tl = list(tl)
          for l_id in tl: label2cnt[l_id] += 1
          text = " ".join(tokenize(reuters.raw(doc_id))) 
          line = text 
          for l_id in tl:  line += "\t" + "__label__" + str(l_id)
          f.write(line+"\n")
  print(data_type)
  for i in range(25): print(i, label2cnt[i])

def load_hclf_data(data_type="train", allow_internal=True, test_true_label=False, in_hierarchy=True):
  layer_1 = ["hier1",  "hier2", "hier3"]
  layer_2 = ["grain", "crude", "livestock", "veg-oil", "meal-feed", "strategic-metal"]
  layer_3 = ["corn",  "wheat", "ship", "nat-gas", "carcass", "hog", "oilseed", "palm-oil", "barley", "rice", "cocoa", "copper", "tin", "iron-steel"]

  label2id = {
              "grain":3, "crude":4, "livestock":10, "veg-oil":11, "meal-feed":17, "strategic-metal":21, 
              "corn":5,  "wheat":6, "ship":7, "nat-gas":8, 
              "carcass":12, "hog":13, "oilseed":14, "palm-oil":15, 
              "barley":18, "rice":19, "cocoa":20, "copper":22, 
              "tin":23, "iron-steel":24
             }
  
  tree_labels = set(layer_2 + layer_3)
  sl_labels   = set(layer_2)
  leaf_labels = set(layer_3)

  # 23
  seqs    = [
             [2,3], [2,4], [9,10], [9,11], [16,17], [16,21],
             [2,3,5], [2,3,6], [2,4,7], [2,4,8], [9,10,12], [9,10,13], [9,11,14], 
             [9,11,15], [16,17,18], [16,17,19], [16,17,20], [16,21,22], [16,21,23], [16,21,24]
            ]

  targets = [
             [2,3,25,0], [2,4,25,0], [9,10,25,0], [9,11,25,0], [16,17,25,0], [16,21,25,0],
             [2,3,5,25], [2,3,6,25], [2,4,7,25], [2,4,8,25], [9,10,12,25], [9,10,13,25], [9,11,14,25], 
             [9,11,15,25], [16,17,18,25], [16,17,19,25], [16,17,20,25], [16,21,22,25], [16,21,23,25], [16,21,24,25]
            ]

  d_inputs = [
             [2,3,0], [2,4,0], [9,10,0], [9,11,0], [16,17,0], [16,21,0],
             [2,3,5], [2,3,6], [2,4,7], [2,4,8], [9,10,12], [9,10,13], [9,11,14], 
             [9,11,15], [16,17,18], [16,17,19], [16,17,20], [16,21,22], [16,21,23], [16,21,24]
            ]

  tree_1 = set([3,4,5,6,7,8])
  tree_2 = set([10,11,12,13,14,15])
  tree_3 = set([17,18,19,20,21,22,23,24])
   
  
  docs = []
  label_seqs = []
  decode_inp = []
  seq_len = []

  def _process(doc_id, seq, target, d_input):
      docs.append(doc_id)
      label_seqs.append(target)
      decode_inp.append([1] +d_input)
      seq_len.append(len(seq)+1)


  mlb = MultiLabelBinarizer()
  documents = reuters.fileids()
  documents = list(filter(lambda doc: doc.startswith(data_type),  documents))

  cnt = 0
  check_cnt = 0
  for doc_id in documents:
      doc_labels = reuters.categories(doc_id)
      dl_seqs = []
      true_labels = []
      for doc_label in doc_labels:
          if doc_label in leaf_labels:
              true_labels.append(label2id[doc_label])
              for seq, target, d_input in list(zip(seqs, targets, d_inputs)):
                  if label2id[doc_label] in seq:
                      dl_seqs.append(seq)
                      if data_type=="train": _process(doc_id, seq, target, d_input)
          if doc_label in sl_labels:
              true_labels.append(label2id[doc_label])
      aug_labels = []
      for true_label in true_labels:
          if true_label in tree_1 and 2 not in aug_labels:  aug_labels.append(2)
          if true_label in tree_2 and 9 not in aug_labels:  aug_labels.append(9)
          if true_label in tree_3 and 16 not in aug_labels:  aug_labels.append(16)
      true_labels += aug_labels 
      for doc_label in doc_labels:
          if not allow_internal: break 
          if doc_label in sl_labels:
              is_sl = True
              for dl_seq in dl_seqs:
                  if label2id[doc_label] in dl_seq:
                      is_sl = False
              if is_sl:
                  for seq, target, d_input in list(zip(seqs, targets, d_inputs)):
                      if label2id[doc_label] in seq and len(seq)==2:
                          dl_seqs.append(seq)
                          if data_type=="train": _process(doc_id, seq, target, d_input)
      if len(dl_seqs)>0:
          #print(data_type, "multi_size:", len(dl_seqs)) 
          cnt += 1
          if data_type=="test" and test_true_label:  
              label_seqs.append(true_labels)
              docs.append(doc_id)
              decode_inp.append([1,0,0,0])   # 1 start 
              seq_len.append(1)
               
          if data_type=="test" and not test_true_label:  
              test_labels = set()
              for dl_seq in dl_seqs:
                  test_labels = test_labels | set(dl_seq)
              test_labels = list(test_labels)
              # if cnt<10: print("test_labels:", test_labels, true_labels)
              label_seqs.append(test_labels)
              docs.append(doc_id)
              decode_inp.append([1,0,0,0])   # 1 start 
              seq_len.append(1)
          # print(doc_id, len(dl_seqs), dl_seqs)
  print(data_type, cnt, len(docs))
  if data_type=="test":  label_seqs = mlb.fit_transform(label_seqs)
  return docs, label_seqs, decode_inp, seq_len                 

def prediction_with_threshold(t_preds, t_scores, threshold):
  t_preds[t_preds == -1] = 0                               
  new_preds = []                                                         
  t_preds = t_preds.transpose((0, 2, 1))
  for i in range(t_preds.shape[0]):
      single = []
      for j in range(t_preds.shape[1]):
          if t_scores[i, j] > threshold or j==0:
              single += t_preds[i, j].tolist()
      new_preds.append(single)
  return new_preds

def test():
  import cli
  config = cli.config
  word2idx = Counter(json.load(open("data/word2idx.json", "r"))["word2idx"])
  train_data = read_data(config, "train", word2idx)
  for batch in train_data.get_batches(60, shuffle=True, cluster=True):
    idxs, ds = batch
    for key, value in ds.data.items():
      print(key, value[0])
      break
    break

  dev_data = read_data(config, "test", word2idx)
  for batch in dev_data.get_batches(60, shuffle=True, cluster=True):
    idxs, ds = batch
    for key, value in ds.data.items():
      print(key, value[0])
      break
    break


if __name__=="__main__":
   trainf = open("data/train.txt", "w")
   get_fasttext(trainf, "train")
   testf = open("data/test.txt", "w")
   get_fasttext(testf, "test")
