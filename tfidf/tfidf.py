__author__ = 'zhuangli'
#! /usr/bin/env python3
from cnn_text_classification import data_helpers
from gensim import corpora, models, similarities
from collections import defaultdict

def get_tfidf_sim(query,docs):
    texts = [[word for word in document.lower().split()] for document in docs]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    vec = dictionary.doc2bow(query.split())
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.token2id))
    sims = index[tfidf[vec]]
    return sims