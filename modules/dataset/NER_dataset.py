from transformers import AutoTokenizer
from torch.utils.data import Dataset,DataLoader
import torch
from collections import OrderedDict
from typing import List
import pickle
import numpy as np

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

class Dataset_NER(Dataset):
    def __init__(self,tag2idx,sentences,labels,tokenizer_path="",do_lower_case=True,max_seq_length=256):
        self.tag2idx = tag2idx
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,do_lower_case=do_lower_case)
        self.max_seq_length = max_seq_length
    def __len__(self):
        return len(self.sentences)
    
    def tokenize(self, words: List[str]):
        """ tokenize input"""
        tokens = []
        valid_positions = []
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word, padding="max_length", max_length=self.max_seq_length, truncation=True)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def processing(self, words:str):
        tokens, valid_positions = self.tokenize(words)
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
        sentence = self.sentences[idx]
        # convert to int
        label = []
        for x in self.labels[idx]:
            if x in self.tag2idx.keys():
                label.append(self.tag2idx[x])
            else:
                label.append(self.tag2idx['O'])
        input_ids,input_mask,segment_ids,valid_ids = self.processing(sentence)
        return input_ids,len(input_ids),input_mask,segment_ids,valid_ids,label,self.tag2idx['O']

def pad(batch):
    '''Pads to the longest sample'''
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = get_element(1)
    maxlen = np.array(seq_len).max()
    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    input_ids = do_pad(0, maxlen)
    input_mask = [[(i>0) for i in ids] for ids in input_ids] 
    LT = torch.LongTensor
    do_pad_label = lambda x, seqlen: [sample[x] + [sample[-1]] * (seqlen - len(sample[x])) for sample in batch] # O
    label = do_pad_label(5, maxlen)
    # sort the index, attn mask and labels on token length
    token_ids = get_element(0)
    token_ids_len = torch.LongTensor(list(map(len, token_ids)))
    _, sorted_idx = token_ids_len.sort(0, descending=True)

    segment_ids = do_pad(3, maxlen)
    valid_ids = do_pad(4, maxlen)

    input_ids = LT(input_ids)[sorted_idx]
    input_mask = LT(input_mask)[sorted_idx]
    segment_ids = LT(segment_ids)[sorted_idx]
    valid_ids = LT(valid_ids)[sorted_idx]
    labels = LT(label)[sorted_idx]
    return input_ids, input_mask, segment_ids, valid_ids, labels, list(sorted_idx.cpu().numpy())

def dataset_train(config, bert_tokenizer="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", do_lower_case=True,max_seq_length=512):
    training_data, validation_data = config.data_dir+config.training_data, config.data_dir+config.val_data 
    train_sentences, train_labels, label_set = read_file_conll(training_data, delim='\t')
    label_set.append('[CLS]')
    label_set.append('[SEP]')
    tag2idx = {t:i for i, t in enumerate(label_set)}
    #print('Training datas: ', len(train_sentences))
    train_dataset = Dataset_NER(tag2idx, train_sentences, train_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case,max_seq_length=512)
    # save the tag2indx dictionary. Will be used while prediction
    with open(config.apr_dir + 'tag2idx.pkl', 'wb') as f:
        pickle.dump(tag2idx, f, pickle.HIGHEST_PROTOCOL)
    dev_sentences, dev_labels, _ = read_file_conll(validation_data, delim='\t')
    dev_dataset = Dataset_NER(tag2idx, dev_sentences, dev_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case,max_seq_length=512)

    #print(len(train_dataset))
    train_iter = DataLoader(dataset=train_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=pad)
    eval_iter = DataLoader(dataset=dev_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    return train_iter, eval_iter, tag2idx


def dataset_test(config, tag2idx, bert_tokenizer="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", do_lower_case=True,max_seq_length=512):
    test_data = config.data_dir+config.test_data
    test_sentences, test_labels, _ = corpus_reader(test_data, delim='\t')
    test_dataset = Dataset_NER(tag2idx, test_sentences, test_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case,max_seq_length=512)
    test_iter = DataLoader(dataset=test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    return test_iter
