#!/usr/bin/env python


"""
Problem Definition :

This script implements KNN algorithm which provides methods to find k closest documents to the given document.

"""

__author__ = 'vivek'

from tfidf import *


class KNN(object):

    def __init__(self, docs):
        self.docs = docs

    def find_k_neighbours(self, target, k):
        """
        Find K nearest neighbours of given doc
        :param docs: list of docs
        :param target: source doc
        :param k: parameter k
        :return: list of k nearest docs.
        """
        distance_list = list()

        # for each doc, find the similarity and update the distance list.
        docs = self.docs
        if target in docs:
            docs.remove(target)

        for i in range(len(docs)):
            doc = docs[i]
            distance_list.append((i, cosine_similarity(doc, target)))

        # sort the list and pick top k results.
        sorted_dist_list = sorted(distance_list, key=lambda x: x[1], reverse=True)

        k_neighbours = list()

        for i in range(k):
            k_neighbours.append(docs[sorted_dist_list[i][0]])

        return k_neighbours


def euclidean_distance(doc1, doc2):
    """
    The euclidean distance between two docs
    :param doc1: First doc
    :param doc2: Second doc
    :return: the distance between docs.
    """

    distance = 0
    v1, v2 = doc1.vector, doc2.vector
    features = list(set(v1.keys()).union(v2.keys()))

    for feature in features:
        distance += pow((v1[feature] - v2[feature]), 2)

    return math.sqrt(distance)


def cosine_similarity(doc1, doc2):
    """
    The cosine_similarity between two docs
    :param doc1: First doc
    :param doc2: Second doc
    :return: the cosine_similarity between docs.
    """

    distance = 0
    v1, v2 = doc1.vector, doc2.vector

    # Choose the doc with less features to lessen the calculations.
    if len(v2.keys()) < len(v1.keys()):
        v1, v2 = v2, v1

    for feature in v1.keys():
        distance += (v1[feature] * v2[feature])

    return distance

