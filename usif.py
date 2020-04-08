#-*- coding:utf-8 -*- coding

import json
import numpy as np
from sklearn.decomposition import TruncatedSVD
from gensim.models import word2vec
import torch
from torchtext.vocab import Vectors

class word2prob(object):

    def __init__(self,word_count_file_path):
        file_path = word_count_file_path
        with open(file_path, 'r') as f:
            self.prob = json.load(f)
            total = sum(self.prob.values())

        self.prob = {k: (self.prob[k] / total) for k in self.prob}
        self.min_prob = min(self.prob.values())
        self.count = total

    def __getitem__(self, w):
        return self.prob.get(w.lower(), self.min_prob)

    def __contains__(self, w):
        return w.lower() in self.prob

    def __len__(self):
        return len(self.prob)

    def vocab(self):
        return iter(self.prob.keys())

class my_word2vec(object):

    def __init__(self,embedding_path):

        self.embedding_path = embedding_path
        vectors = Vectors(name=embedding_path)
        self.vectors = vectors

    def __getitem__(self, w):
        return self.vectors[w]

    def __contains__(self, w):
        return w in self.vectors


class sent_uSIF(object):
    """Embed sentences using unsupervised smoothed inverse frequency."""

    def __init__(self, vec, prob, n=11, m=1):

        self.vec = vec
        self.m = m

        if not (isinstance(n, int) and n > 0):
            raise TypeError("n should be a positive integer")

        vocab_size = float(len(prob))
        threshold = 1 - (1 - 1 / vocab_size) ** n
        alpha = len([w for w in prob.vocab() if prob[w] > threshold]) / vocab_size
        Z = 0.5 * vocab_size
        self.a = (1 - alpha) / (alpha * Z)

        self.weight = lambda word: (self.a / (0.5 * self.a + prob[word]))

    def _to_vec(self, sentence):


        tokens = sentence.split(' ')


        if tokens == []:
            return np.zeros(300) + self.a
        else:
            v_t = np.array([self.vec[t].numpy() for t in tokens])
            v_t = v_t * (1.0 / np.linalg.norm(v_t, axis=0))
            v_t = np.array([self.weight(t) * v_t[i, :] for i, t in enumerate(tokens)])
            return np.mean(v_t, axis=0)

    def embed(self, sentences):

        vectors = [self._to_vec(s) for s in sentences]

        if self.m == 0:
            return vectors

        proj = lambda a, b: a.dot(b.transpose()) * b
        svd = TruncatedSVD(n_components=self.m, random_state=0).fit(vectors)

        for i in range(self.m):
            lambda_i = (svd.singular_values_[i] ** 2) / (svd.singular_values_ ** 2).sum()
            pc = svd.components_[i]
            vectors = [v_s - lambda_i * proj(v_s, pc) for v_s in vectors]

        return np.array(vectors)

