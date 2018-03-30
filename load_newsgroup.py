from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups as fetch_data
from sklearn.preprocessing import MultiLabelBinarizer
from load_data import DataSet, tokenize

def read_data(config, data_type="train", word2idx=None, max_seq_length=3):
  print("preparing {} data".format(data_type))
  docs, label_seqs, decode_inps, seq_lens = load_hclf(config, data_type=data_type) 
  docs = [tokenize(reuters.raw(doc_id)) for doc_id in docs]
  for doc in docs:
    # print(len(doc))
    config.max_docs_length = len(doc) if len(doc) > config.max_docs_length else config.max_docs_length

  print("max_doc_length:", data_type, config.max_docs_length)
  docs2mat = [[word2idx[doc[_]] if _ < len(doc) else 1 for _ in range(config.max_docs_length)] for doc in docs] 
  docs2mask = [[1 if _ < len(doc) else 0 for _ in range(config.max_docs_length)] for doc in docs] 
  
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

def load_hclf(config, data_type = data_type):
  seqs = {
          [22,3], [22,4], [22,5], [22,6], [22,7],
          [23,9], [23,10], [23,11], [23,12], 
          [24,13], [24,14], [24,15], [24,16], 
          [25,8], 
          [26,20], [26,18], [26,19], 
          [27,21], [27,2], [27,17]
         }
  news = fetch_data(subset=data_type, remove=('headers', 'footers', 'quotes'))
  docs = news.data
  labels = news.target + 2

  label_seqs = []
  decode_inp = []
  y = []
  seq_len = []
  mlb = MultiLabelBinarizer()
  mlb.fit([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])  
  
  for label in labels:
    for seq in seqs:
      if label in seq: 
        y += [seq]
        label_seqs += [seq+[28]]
        decode_inp += [[1]+seq]
        seq_len += [3]
  print(data_type, len(docs))
  if data_type=="test" or config.model_name=="RCNN_flat":  label_seqs = mlb.fit_transform(y)
  return docs, label_seqs, decode_inp, seq_len                 



def main():
  pass

if __name__=="__main__":
  main()
