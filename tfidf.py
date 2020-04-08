#-*- coding:utf-8 -*- coding

import math
from collections import Counter

def tf(word, sentence_count):
    return sentence_count[word] / sum(sentence_count.values())

def n_containing(word, sentences_count):
    return sum(1 for sentence in sentences_count if word in sentence)

def idf(word, sentences_count):

    return math.log(len(sentences_count) / (1+n_containing(word, sentences_count)))

def tfidf(word, sentence_count, sentences_count):
    return tf(word, sentence_count)*idf(word, sentences_count)

class sent_tfidf(object):
    def __init__(self,vec):

        self.vec = vec

    def embed(self,sentences_list):
        sentences_count = [Counter(item.split()) for item in sentences_list]
        vectors = []
        for i, sentence_count in enumerate(sentences_count):
            scores = {word: tfidf(word, sentence_count, sentences_count) for word in sentence_count}
            sentence_vector = sum([self.vec[word].numpy() * scores[word] for word in sentences_list[i].split()])
            vectors.append(sentence_vector)

        return vectors


class sent_average(object):
    def __init__(self,vec):
        self.vec = vec

    def embed(self,sentences_list):
        vectors = []
        for i, sentence in enumerate(sentences_list):

            sentence_vector = sum([self.vec[word].numpy()  for word in sentences_list[i].split()])
            vectors.append(sentence_vector)

        return vectors












