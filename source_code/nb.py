#!/usr/bin/env python


"""
Problem Definition :

This script is an implementation of Multinomial Naive bayes.

"""

__author__ = 'vivek'

import operator
import math
from collections import defaultdict


class NaiveBayes(object):

    def __init__(self):
        # stores (class, #documents)
        self.class_doc_count = defaultdict(lambda: 0)
        # stores (class, prior)
        self.class_priors = defaultdict(lambda: 0)
        # stores vocab count of entire data-set
        self.vocab_count = None
        # stores (class,(term, freq)
        self.class_term_freq = defaultdict(lambda: defaultdict(lambda: 0))
        # stores (class, token_count)
        self.class_token_count = defaultdict(lambda: 0)

        self.confusion_matrix = None
        self.stats = None

    def train(self, documents):
        """
        Given a list of labeled Document objects, compute the class priors and class feature stats.
        """

        vocab_list = list()

        for document in documents:

            topic = document.topic
            tf = document.tf

            # count documents per class
            self.class_doc_count[topic] += 1

            for term in tf.keys():
                self.class_term_freq[topic][term] = self.class_term_freq[topic][term] + 1
                self.class_token_count[topic] = self.class_token_count[topic] + 1
                vocab_list.append(term)

        self.vocab_count = len(set(vocab_list))

        for key in self.class_doc_count.keys():
            self.class_priors[key] = float(self.class_doc_count[key])/float(len(documents))

    def classify(self, documents):
        """
        Classify the list of documents.
        :param documents: The list of documents.
        :return: a list of strings, the class topics, for each document.
        """

        predictions = list()
        scores = defaultdict(lambda: 0)

        for document in documents:
            tf = document.tf

            for topic, prior in self.class_priors.items():
                scores[topic] = math.log10(prior)

                for token in tf.keys():
                    token_score = tf[token] * math.log10((self.class_term_freq[topic][token] + 1)*1.0/
                                                         (self.class_token_count[topic] + self.vocab_count))
                    scores[topic] += token_score

            predictions.append(max(scores.iteritems(), key=operator.itemgetter(1))[0])

            scores = dict.fromkeys(scores, 0)

        return predictions

