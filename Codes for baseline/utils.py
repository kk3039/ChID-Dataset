import random
import pickle
import os
import re
import json
import jieba
import time
import numpy as np


class Vocabulary:
    def __init__(self):
        with open("/Users/yinglu/Documents/grad_school/nlp/ChID-Dataset/Codes for baseline/cloth_candidate_dict2.txt") as f:
            id2idiom = re.findall(r'<\w+>|\w+', f.readline())
            id2idiom = list(dict.fromkeys(id2idiom))

        self.id2idiom = ["<PAD>","<UNK>"] + id2idiom
        self.idiom2id = {}
        
        for id, idiom in enumerate(self.id2idiom):
            self.idiom2id[idiom] = id

        # with open("wordList.txt") as f:
        with open("/Users/yinglu/Documents/grad_school/nlp/ChID-Dataset/Codes for baseline/cloth_word_dict2.txt") as f:
            id2word = re.findall(r'<\w+>|\w+', f.readline())
            id2word = list(dict.fromkeys(id2word))
        self.id2word = ["<PAD>", "<UNK>", "#idiom#"] + id2word
        self.word2id = {}
        for id, word in enumerate(self.id2word):
            self.word2id[word] = id
        self.unknown_candidate = 0
        self.unknown_word = 0
        self.total_words = 0
        print("id2idiom %d, idiom2id %d, id2word %d, word2id %d" % (len(self.id2idiom) ,len(self.idiom2id), len(self.id2word), len(self.word2id)))

    def return_unknown_index(self):
        return self.idiom2id["<UNK>"], self.word2id["<UNK>"]

    def return_unknown_counts(self):
        return self.unknown_candidate, self.unknown_word, self.total_words
        


    def tran2id(self, token, is_idiom=False):
        self.total_words += 1 
        if is_idiom:
            if token in self.idiom2id:
                return self.idiom2id[token]
            else:
                self.unknown_candidate +=1
                return self.idiom2id["<UNK>"]
        else:
            if token in self.word2id:
                return self.word2id[token]
            else:
                self.unknown_word +=1
                return self.word2id["<UNK>"]



def caculate_acc(original_labels, pred_labels):

    acc_blank = np.zeros((2, 2), dtype=np.float32)
    acc_array = np.zeros((2), dtype=np.float32)

    for id in range(len(original_labels)): # batch_size
        ori_label = original_labels[id]
        pre_label = list(pred_labels[id])

        x_index = 0 if len(ori_label) == 1 else 1

        for real, pred in zip(ori_label, pre_label):

            acc_array[1] += 1
            acc_blank[x_index, 1] += 1

            if real == pred:
                acc_array[0] += 1
                acc_blank[x_index, 0] += 1

    return acc_array, acc_blank
