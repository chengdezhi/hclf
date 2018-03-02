from nltk.corpus import reuters 

layer_1 = ["hier1",  "hier2", "hier3"]
layer_2 = ["grain", "crude", "livestock", "veg-oil", "meal-feed", "strategic-metal"]
layer_3 = ["corn",  "wheat", "ship", "nat-gas", "carcass", "hog", "oilseed", "palm-oil", "barley", "rice", "cocoa", "copper", "tin", "iron-steel"]

h_label_set = layer_1 + layer_2 + layer_3 


def get_label2id(labels):
    label2id = {}
    for idx,label in enumerate(labels):
        label2id[label] = idx
    return label2id
id2label = layer_1 + layer_2 + layer_3 
label2id = get_label2id(id2label)  # {"hier1":0,...}
h_parent = {"corn":"grain", "wheat":"grain", "ship":"crude", "nat-gas":"crude", "carcass":"livestock"}


sub_category = layer_2 + layer_3 

def check_hclf():
    documents = reuters.fileids()
    categories = reuters.categories()
    print("cat:", categories)
    for label in sub_category:
        assert label in categories, label
        
        cat_docs = reuters.fileids(label)
        if label in h_parent:
            for doc_id in cat_docs:
                print(reuters.categories(doc_id))

        train_doc =  list(filter(lambda doc: doc.startswith("train"),  cat_docs))
        test_doc  =  list(filter(lambda doc: doc.startswith("test"),  cat_docs))
        # print(label, len(cat_docs), len(train_doc), len(test_doc))
    train_docs, test_docs = list(), list()
    for label in sub_category:
        cat_docs = reuters.fileids(label)
        train_docs +=  list(filter(lambda doc: doc.startswith("train"),  cat_docs))
        test_docs  +=  list(filter(lambda doc: doc.startswith("test"),  cat_docs))
    
    print("train:", len(train_docs))
    print("test:",  len(test_docs))

 
def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents")
 
    train_docs = list(filter(lambda doc: doc.startswith("train"),
                        documents))
    print(train_docs[0:3])
    print(str(len(train_docs)) + " total train documents")
 
    test_docs = list(filter(lambda doc: doc.startswith("test"),
                       documents))
    print(str(len(test_docs)) + " total test documents")
 
    # List of categories
    categories = reuters.categories()
    print(str(len(categories)) + " categories");
    print("categories:", categories) 
    # Documents in a category
    category_docs = reuters.fileids("acq");
 
    # Words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0]);
    print(document_words);  
    print(len(document_words));  
 
    # Raw document
    print(reuters.raw(document_id));


# collection_stats()
check_hclf()
