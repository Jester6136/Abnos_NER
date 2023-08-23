from transformers import AutoTokenizer
from torch.utils.data import Dataset,DataLoader
import torch
from collections import OrderedDict
from typing import List
import pickle

def read_file_conll(path, delim='\t', word_idx=0, label_idx=-1):
    tokens, labels = [], []
    tmp_tok, tmp_lab = [], []
    label_set = []
    with open(path, 'r') as reader:
        for line in reader:
            line = line.strip()
            cols = line.split(delim)
            if len(cols) < 2:
                if len(tmp_tok) > 0:
                    tokens.append(tmp_tok); labels.append(tmp_lab)
                tmp_tok = []
                tmp_lab = []
            else:
                tmp_tok.append(cols[word_idx])
                tmp_lab.append(cols[label_idx])
                label_set.append(cols[label_idx])
    return tokens, labels, list(OrderedDict.fromkeys(label_set))

class NER_dataset(Dataset):
    def __init__(self,tag2idx,sentences,labels,tokenizer_path="",do_lower=True,max_seq_length=512):
        self.tag2idx = tag2idx
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = AutoTokenizer(tokenizer_path,do_lower_case=do_lower)
        self.max_seq_length = max_seq_length
    def __len__(self):
        return len(self.sentences)
    
    def tokenize(self, words: List[str]):
        """ tokenize input"""
        tokens = []
        valid_positions = []
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def processing(self,words:str):
        tokens, valid_positions = tokenize(words)
        tokens.insert(0,"[CLS]")
        valid_positions.insert(0,1)
        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids,input_mask,segment_ids,valid_positions

    def __getitem__(self,idx):
        sentence = sentences[idx]
        label = []
        for x in self.labels[idx]:
            if x in self.tag2idx.keys():
                label.append(self.tag2idx[x])
            else:
                label.append(self.tag2idx['O'])
        
        input_ids,input_mask,segment_ids,valid_ids = self.preprocess(sentence)
        return input_ids,input_mask,segment_ids,valid_ids,label

def dataset_train(config, bert_tokenizer="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", do_lower_case=True,max_seq_length=512):
    training_data, validation_data = config.data_dir+config.training_data, config.data_dir+config.val_data 
    train_sentences, train_labels, label_set = read_file_conll(training_data, delim='\t')
    tag2idx = {t:i for i, t in enumerate(label_set)}
    #print('Training datas: ', len(train_sentences))
    train_dataset = NER_Dataset(tag2idx, train_sentences, train_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case,max_seq_length=512)
    # save the tag2indx dictionary. Will be used while prediction
    with open(config.apr_dir + 'tag2idx.pkl', 'wb') as f:
        pickle.dump(tag2idx, f, pickle.HIGHEST_PROTOCOL)
    dev_sentences, dev_labels, _ = read_file_conll(validation_data, delim='\t')
    dev_dataset = NER_Dataset(tag2idx, dev_sentences, dev_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case,max_seq_length=512)

    #print(len(train_dataset))
    train_iter = DataLoader(dataset=train_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=4)
    eval_iter = DataLoader(dataset=dev_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1)
    return train_iter, eval_iter, tag2idx


def dataset_test(config, tag2idx, bert_tokenizer="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", do_lower_case=True,max_seq_length=512):
    test_data = config.data_dir+config.test_data
    test_sentences, test_labels, _ = corpus_reader(test_data, delim='\t')
    test_dataset = NER_Dataset(tag2idx, test_sentences, test_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case,max_seq_length=512)
    test_iter = DataLoader(dataset=test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1)
    return test_iter