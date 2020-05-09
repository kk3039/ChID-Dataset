# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import random
import re
import time
# import jieba
from utils import Vocabulary
import nltk
import string
import json

random.seed(time.time())


class DataManager:
    def __init__(self):
        # jieba.load_userdict("dict.txt")
        if not os.path.exists('cache'):
            os.makedirs('cache')

        if os.path.exists("cache/vocab.pkl"):
            self.vocab = pickle.load(open("cache/vocab.pkl", "rb"))
        else:
            self.vocab = Vocabulary()
            pickle.dump(self.vocab, open("cache/vocab.pkl", "wb"), protocol=2)

        print("*** Finish building vocabulary")


    def get_num(self):
        num_word, num_idiom = len(self.vocab.id2word) - 3, len(self.vocab.id2idiom) - 1
        print("Numbers of words and idioms: %d %d" % (num_word, num_idiom))
        return num_word, num_idiom

    def _tokenize(self, st, sentence_split=None, option=False):
        #TODO: The tokenizer's performance is suboptimal
        if option and (st[-1] in string.punctuation or st[0] == "$"): #to deal with options
            st = st[:-1]
        if len(st) > 0:
            if option and (st[0] in string.punctuation or st[0] == "$"):  # to deal with options
                st = st[0:]
        st = st.replace("<IMG>", "")
        st = st.replace("[KS5UKS5U]", "")
        st = st.replace("[:Z|xx|k.Com]", "")
        st = st.replace("(;)", "")
        ans = []
        for sent in nltk.sent_tokenize(st):
            if sentence_split is not None and len(ans) > 0:
                ans += [sentence_split]
            for w in nltk.word_tokenize(sent):
                w = w.lower()
                if len(ans) > 0 and (w == "'re" or w == "n't" or w == "'s" or w == "'m" or w == "'" and len(ans[-1]) > 0) and ans[-1] != "_":
                    ans[-1] += w
                else:
                    ans += [w]

        if option and ans.find(" ") != -1:
            # print ans
            print (ans)
        return ans

    def _prepare_data(self, temp_data):
        cans = temp_data["options"]
        answers = temp_data["answers"]
        labs = []
        for a in answers:
            a = a.upper()
            labs.append(ord(a)-ord("A"))
        cans = [[self.vocab.tran2id(each.lower(), True) for each in each_cans] for each_cans in cans]

        content = temp_data["article"].replace(".", " . ")
        content = self._tokenize(content)
        doc = []
        loc = []
        
        for i, token in enumerate(content):
            doc.append(self.vocab.tran2id(token))
            if token == "_":
                loc.append(i)

        assert len(loc) == len(labs) == content.count("_")
        unknown_candidate, unknown_word, total_words = self.vocab.return_unknown_counts()
        return doc, cans, labs, loc, unknown_candidate, unknown_word, total_words

    def train(self):
        difficulty_set = ["middle", "high"]
        data_dir = "../CLOTH/"
        for d in difficulty_set:
            new_path = os.path.join(data_dir, "train", d)
            for inf in os.listdir(new_path):
                inf_path = os.path.join(new_path, inf)
                obj = json.load(open(inf_path, "r"))

                doc, cans, labs, loc, unknown_candidate, unknown_word, total_words = self._prepare_data(obj)
                del obj
                yield doc, cans, labs, loc, unknown_candidate, unknown_word, total_words


    def valid(self, mode="dev"): # "dev" or "test" or "out"
        difficulty_set = ["middle", "high"]
        data_dir = "../CLOTH/"

        for d in difficulty_set:
            new_path = os.path.join(data_dir, "test", d)
            for inf in os.listdir(new_path):
                inf_path = os.path.join(new_path, inf)
                obj = json.load(open(inf_path, "r"))

                doc, cans, labs, loc, unknown_candidate, unknown_word, total_words = self._prepare_data(obj)
                del obj
                yield doc, cans, labs, loc, unknown_candidate, unknown_word, total_words
    def return_unknown_words(self):
        return self.vocab

    def get_embed_matrix(self):  # DataManager
        if os.path.exists("cache/word_embed_matrix.npy") and os.path.exists("cache/idiom_embed_matrix.npy"):
            self.word_embed_matrix = np.load("cache/word_embed_matrix.npy")
            self.idiom_embed_matrix = np.load("cache/idiom_embed_matrix.npy")

        else:
            np.random.seed(37)
            def embed_matrix(file, dic, dim=200):
                fr = open(file, encoding="utf8")
                wv = {}
                for line in fr:
                    vec = line.split(" ")
                    word = vec[0]
                    if word in dic:
                        vec = [float(value) for value in vec[1:]]
                        assert len(vec) == dim
                        wv[dic[word]] = vec
                        # which indicates the order filling in wv is the same as id2idiom/id2word

                lost_cnt = 0
                matrix = []
                for i in range(len(dic)):
                    if i in wv:
                        matrix.append(wv[i])
                    else:
                        lost_cnt += 1
                        matrix.append(np.random.uniform(-0.1, 0.1, [dim]))
                print("dic %d, matrix %d, lost_cnt %d" % (len(dic), len(matrix), lost_cnt))
                return matrix, lost_cnt

            # self.word_embed_matrix, lost_word = embed_matrix("../data/wordvector.txt", self.vocab.word2id)
            # self.idiom_embed_matrix, lost_idiom = embed_matrix("../data/idiomvector.txt", self.vocab.idiom2id)
            
            self.word_embed_matrix, lost_word = embed_matrix("cloth_word_vectors2.txt", self.vocab.word2id)
            self.idiom_embed_matrix, lost_idiom = embed_matrix("cloth_candidate_vectors2.txt", self.vocab.idiom2id)
            self.word_embed_matrix = np.array(self.word_embed_matrix, dtype=np.float32)
            self.idiom_embed_matrix = np.array(self.idiom_embed_matrix, dtype=np.float32)
            np.save("cache/word_embed_matrix.npy", self.word_embed_matrix)
            np.save("cache/idiom_embed_matrix.npy", self.idiom_embed_matrix)
            print("*** %d idioms and %d words not found" % (lost_idiom, lost_word))

        print("*** Embed matrixs built")
        return self.word_embed_matrix, self.idiom_embed_matrix
