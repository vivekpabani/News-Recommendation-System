#!/usr/bin/env python


"""
Problem Definition :

This script is an implementation of rank classifier which depends on the tfidfie indexing method.

"""

__author__ = 'vivek'

from collections import Counter, defaultdict
import operator
import math


class RankClassifier(object):

    def __init__(self):
        self.train_docs = None
        self.topic_list = None
        self.topic_set = None
        self.index_dict = None
        self.confusion_matrix = None
        self.stats = None

    def train(self, train_docs):
        self.train_docs = train_docs
        self.topic_list = list(set([d.topic for d in self.train_docs]))
        self.topic_set = TopicSet(self.train_docs, self.topic_list)
        self.index_dict = self.create_index_dict()

    def create_index_dict(self):

        topic_train_docs = defaultdict(lambda: list())

        for doc in self.train_docs:
            topic_train_docs[doc.topic].append(doc)

        index_dict = defaultdict(lambda: None)

        for topic, docs in topic_train_docs.items():
            index_dict[topic] = LocalIndex(docs, self.topic_set.text_common_tokens)

        return index_dict

    def classify(self, documents):

        predictions = list()
        for doc in documents:
            score_dict = defaultdict(lambda: 0)
            for topic, index in self.index_dict.items():
                score_dict[topic] = self.cal_score(doc, index)

            predictions.append(max(score_dict.iteritems(), key=operator.itemgetter(1))[0])

        return predictions

    def cal_score(self, doc, index):
        text_tfidf = index.topic_text_tfidf
        title_tfidf = index.topic_title_tfidf

        title_score, text_score = 0.0, 0.0

        text_len = len(doc.text_tokens)
        title_len = len(doc.title_tokens)

        for token in doc.title_tokens:
            if token in title_tfidf:
                title_score += (2*title_tfidf[token])
            elif token in text_tfidf:
                title_score += (1.5*text_tfidf[token])

        for token in doc.text_tokens:
            if token in text_tfidf.keys():
                text_score += text_tfidf[token]

        total_score = (title_score/title_len) + (text_score/text_len)

        return total_score


class TopicSet(object):

    def __init__(self, docs, topics):
        self.docs = docs
        self.topics = topics
        self.title_tokens = defaultdict(lambda: list())
        self.text_tokens = defaultdict(lambda: list())
        self.title_doc_freqs = defaultdict(lambda: list())
        self.text_doc_freqs = defaultdict(lambda: list())

        for doc in self.docs:
            self.title_tokens[doc.topic].append(doc.title_tokens)
            self.text_tokens[doc.topic].append(doc.text_tokens)

        for topic in self.topics:
            self.title_doc_freqs[topic] = self.count_doc_frequencies(self.title_tokens[topic])
            self.text_doc_freqs[topic] = self.count_doc_frequencies(self.text_tokens[topic])

        self.title_common_tokens = self.find_common_tokens(self.title_doc_freqs)
        self.text_common_tokens = self.find_common_tokens(self.text_doc_freqs)

    def count_doc_frequencies(self, token_l):
        """
        :param token_l: A list of lists of tokens, one per document. This is the output of the tokenize method.
        :return: A dict mapping from a term to the number of documents that contain it.
        """

        doc_freqs = defaultdict(lambda: 0)
        doc_count = len(token_l)

        for doc in token_l:
            for token in set(doc):
                doc_freqs[token] += 1

        for key, value in doc_freqs.items():
            doc_freqs[key] = value*1.0 / doc_count

        return doc_freqs

    def find_common_tokens(self, doc_freqs):

        token_count = defaultdict(lambda: 0)
        common_tokens = list()

        threshold = int(math.floor(len(doc_freqs.keys())/2.0))

        for topic in doc_freqs.keys():
            doc_freq = doc_freqs[topic]
            doc_tokens = filter(lambda x: doc_freq[x] > 0.1, doc_freq.keys())

            for token in doc_tokens:
                token_count[token] += 1

        for token, count in token_count.items():
            if count > threshold:
                common_tokens.append(token)

        return common_tokens


class LocalIndex(object):

    def __init__(self, docs, text_exclude_tokens=None, title_exclude_tokens=None):

        if not text_exclude_tokens:
            text_exclude_tokens = list()
        if not title_exclude_tokens:
            title_exclude_tokens = list()

        self.docs = docs

        self.title_tokens = [[t for t in d.title_tokens if t not in (title_exclude_tokens+text_exclude_tokens)] for d in self.docs]
        self.title_doc_freqs = self.count_doc_frequencies(self.title_tokens)
        self.title_index = self.create_tf_index(self.title_tokens)
        self.title_lengths, self.mean_title_length = self.compute_doc_lengths(self.title_tokens)
        self.title_tfidf = self.create_tfidf_index(self.title_tokens, self.title_index, self.title_lengths, self.title_doc_freqs)
        self.topic_title_tfidf = self.create_topic_tfidf_index(self.title_tokens, self.title_index, self.title_doc_freqs)

        self.text_tokens = [[t for t in d.text_tokens if t not in text_exclude_tokens] for d in self.docs]
        self.text_doc_freqs = self.count_doc_frequencies(self.text_tokens)
        self.text_index = self.create_tf_index(self.text_tokens)
        self.text_lengths, self.mean_text_length = self.compute_doc_lengths(self.text_tokens)
        self.text_tfidf = self.create_tfidf_index(self.text_tokens, self.text_index, self.text_lengths, self.text_doc_freqs)
        self.topic_text_tfidf = self.create_topic_tfidf_index(self.text_tokens, self.text_index, self.text_doc_freqs)

    def count_doc_frequencies(self, token_l):
        """
        :param token_l: A list of lists of tokens, one per document. This is the output of the tokenize method.
        :return: A dict mapping from a term to the number of documents that contain it.
        """

        doc_freqs = defaultdict(lambda: 0)

        for doc in token_l:
            for token in set(doc):
                doc_freqs[token] += 1

        return doc_freqs

    def create_tf_index(self, token_l):
        """
        Create an index in which each postings list contains a list of [doc_id, tf weight] pairs.
        :param token_l: list of lists, where each sublist contains the tokens for one document.
        :return:
        """

        index = defaultdict(lambda: list())

        for i in range(len(token_l)):
            doc = token_l[i]
            counter = Counter(doc)
            for token in counter.keys():
                index[token].append([i, counter[token]])
        
        return index

    def create_tfidf_index(self, token_l, tf_index, doc_lengths, doc_freqs):

        doc_count = len(token_l)

        tfidf = defaultdict(lambda: list())

        for token, freq_l in tf_index.items():
            token_idf = doc_freqs[token]*1.0/doc_count
            for token_freq in freq_l:
                score = (1 + token_freq[1]*1.0/doc_lengths[token_freq[0]]) * (1 + token_idf)
                tfidf[token_freq[0]].append([token, score])

        return tfidf

    def create_topic_tfidf_index(self, token_l, tf_index, doc_freqs):

        doc_count = len(token_l)
        all_token_count = sum(len(t) for t in token_l)

        tfidf = defaultdict(lambda: 0)

        for token, freq_l in tf_index.items():
            token_count = sum(i[1] for i in freq_l)
            token_idf = doc_freqs[token]*1.0/doc_count
            score = (1 + token_count*1.0/all_token_count) * (1 + token_idf)
            tfidf[token] = score

        return tfidf

    def compute_doc_lengths(self, token_l):

        doc_lengths = defaultdict(lambda: 0)
        total_len = 0

        for i in range(len(token_l)):
            doc = token_l[i]
            doc_len = len(doc)
            doc_lengths[i] = doc_len
            total_len += doc_len

        return doc_lengths, total_len/len(token_l)
