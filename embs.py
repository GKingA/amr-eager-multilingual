#!/usr/bin/env python
#coding=utf-8

'''
Definition of OneHotEncoding class (for named entity sparse representation),
PretrainedEmbs (for pretrained embeddings) and RndInitLearnedEmbs (for random
initialized ones). Embs puts everything together, using the same random seed.

@author: Marco Damonte (m.damonte@sms.ed.ac.uk)
@since: 03-10-16
'''

import re
import string
import random

class OneHotEncoding:
    def __onehot(self, index):
        onehot = [0]*(self.dim)
        onehot[index  - 1] = 1
        return onehot

    def  __init__(self, vocab):
        lines = open(vocab).readlines()
        self.dim = len(lines) + 3
        self.enc = {}
        for counter, line in enumerate(lines):
            self.enc[line.strip()] = self.__onehot(counter + 1)
        self.enc["<TOP>"] = self.__onehot(len(self.enc) + 1)
        self.enc["<NULL>"] = self.__onehot(len(self.enc) + 1)
        self.enc["<UNK>"] = self.__onehot(len(self.enc) + 1)
    def get(self, label):
        assert(label is not None)
        if label == "<TOP>":
            return self.enc["<TOP>"]
        if label.startswith("<NULL"):
            return self.enc["<NULL>"]
        if label in self.enc:
            return self.enc[label]
        return self.enc["<UNK>"]

class PretrainedEmbs:
    def __init__(self, generate, initializationFileIn, initializationFileOut, dim, unk, root, nullemb, prepr, punct):
        self.prepr = prepr
        self.indexes = {}
        self.initialization = {}
        self.counter = 1
        self.dim = dim
        self.punct = punct
        self.nullemb = nullemb
        self.vecs = {}

        if generate:
            fw = open(initializationFileOut, "w")
        for line in open(initializationFileIn).readlines()[2:]: # first two lines are not actual embeddings
            v = line.split()
            word = v[0]
            self.vecs[word] = " ".join(v[1:])
            if self.prepr:
                word = self._preprocess(word)
            if word in self.indexes:
                continue
            self.indexes[word] = self.counter
            if generate:
                fw.write(v[1])
                for i in v[2:]:
                    fw.write("," + str(i))
                fw.write("\n")
            self.counter += 1
        self.indexes["<UNK>"] = self.counter
        if generate:
            fw.write(str(unk[0]))
            for i in unk[1:]:
                fw.write("," + str(i))
            fw.write("\n")
        self.counter += 1

        self.indexes["<TOP>"] = self.counter
        if generate:
            fw.write(str(root[0]))
            for i in root[1:]:
                fw.write("," + str(i))
            fw.write("\n")
        self.counter += 1

        self.indexes["<NULL>"] = self.counter
        if generate:
            fw.write(str(nullemb[0]))
            for i in nullemb[1:]:
                fw.write("," + str(i))
            fw.write("\n")
        self.counter += 1

        if punct is not None:
            self.indexes["<PUNCT>"] = self.counter
            if generate:
                fw.write(str(punct[0]))
                for i in punct[1:]:
                    fw.write("," + str(i))
                fw.write("\n")
            self.counter += 1
        
    def get(self, word):
        assert(word is not None)
        w = word if isinstance(word, str) else word.decode('utf-8')
        if w == "<TOP>":
            return self.indexes["<TOP>"]
        if w.startswith("<NULL"):
            return self.indexes["<NULL>"]

        if self.prepr:
            w = self._preprocess(w)
        if self.punct is not None and w not in self.indexes and w in list(string.punctuation):
            return self.indexes["<PUNCT>"]
        elif w in self.indexes:
            return self.indexes[w]
        else:
            return self.indexes["<UNK>"]

    def _preprocess(self, word):
        w = word if isinstance(word, str) else word.decode('utf-8')
        if w.startswith('"') and w.endswith('"') and len(w) > 2:
            w = w[1:-1]
        reg = re.compile(".+-[0-9][0-9]")
        w = w.strip().lower()
        if reg.match(w) is not None:
            w = w.split("-")[0]
        if re.match("^[0-9]", w) is not None:
            word = word[0]
        w = w.replace("0","zero").replace("1","one").replace("2","two").replace("3","three").replace("4","four")\
             .replace("5","five").replace("6","six").replace("7","seven").replace("8","eight").replace("9","nine")
        return w
        
    def vocabSize(self):
        return self.counter - 1



class RndInitLearnedEmbs:
    def __init__(self, vocab):
        self.indexes = {}
        for counter, line in enumerate(open(vocab)):
            word = line.strip()
            self.indexes[word] = counter + 1
        self.indexes["<UNK>"] = len(self.indexes) + 1
        self.indexes["<TOP>"] = len(self.indexes) + 1
        self.indexes["<NULL>"] = len(self.indexes) + 1

    def get(self, label):
        assert(label is not None and label != "")
        if label == "<TOP>":
            return self.indexes["<TOP>"]
        if label.startswith("<NULL"):
            return self.indexes["<NULL>"]

        if label not in self.indexes:
            label = "<UNK>"
        return self.indexes[label]
        
    def vocabSize(self):
        return len(self.indexes)

class Embs:

    def _create_concept_vec(self, wordembs, conceptembs, propbank, wordvecs):
        fw = open(conceptembs, "w") 
        for line in open(wordembs):
            fw.write(line.strip() + "\n")
        for p in open(propbank):
            p2 = p.split("-")[0] 
            if p2 in wordvecs.indexes:
                fw.write(p.strip() + " " + wordvecs.vecs[p2] + "\n")
        fw.close()
            
    def __init__(self, resources_dir, model_dir, generate = False):
        random.seed(0)
        punct100 = [float(0.02*random.random())-0.01 for i in range(100)]
        unk100 = [float(0.02*random.random())-0.01 for i in range(100)]
        root100 = [float(0.02*random.random())-0.01 for i in range(100)]

        unk50 = [float(0.02*random.random())-0.01 for i in range(50)]
        root50 = [float(0.02*random.random())-0.01 for i in range(50)]

        unk10 = [float(0.02*random.random())-0.01 for i in range(10)]
        root10 = [float(0.02*random.random())-0.01 for i in range(10)]

        null10 = [float(0.02*random.random())-0.01 for i in range(10)]
        null50 = [float(0.02*random.random())-0.01 for i in range(50)]

        punct50 = [float(0.02*random.random())-0.01 for i in range(50)]

        self.deps = RndInitLearnedEmbs(model_dir + "/dependencies.txt")
        self.pos =  RndInitLearnedEmbs(resources_dir + "/postags.txt")
        self.words = PretrainedEmbs(generate, resources_dir + "/wordvec50.txt", resources_dir + "/wordembs.txt", 50, unk50, root50, null50, True, punct50)
        self.nes = OneHotEncoding(resources_dir + "/namedentities.txt")
