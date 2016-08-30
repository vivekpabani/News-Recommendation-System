#!/usr/bin/env python


"""
Problem Definition :

This script creates instances of all the classifiers and trains/tests them. It then provides an interface to the
recommendation system, where user can select a document from 'n' choices, and system recommends 'm' similar documents,
where 'n' and 'm' are user selected.

"""

__author__ = 'vivek'

import time
import os
from nb import NaiveBayes
from rank_classifier import RankClassifier
from knn import KNN
import random
from document import Document
from tfidf import Index
from kmeans import KMeans
from util import *
from collections import defaultdict, Counter


def recommendation(all_docs, test_docs, classifier_list):

    print("Recommendation System")
    print("---------------------")

    # ask user for the desired option count and recommendation count. set default value in case invalid inputs.
    try:
        option_count = int(raw_input("\nEnter number of articles to choose from. [number from 5 to 10 suggested]: "))
        if option_count < 1 or option_count > 20:
            print("Invalid Choice.. By default selected 5.")
            option_count = 5
    except:
        print("Invalid Choice.. By default selected 5.")
        option_count = 5

    try:
        k_n = int(raw_input("\nEnter number of recommendation per article. [number from 5 to 10 suggested]: "))
        if k_n < 1 or k_n > 20:
            print("Invalid Choice.. By default selected 5.")
            k_n = 5
    except:
        print("Invalid Choice.. By default selected 5.")
        k_n = 5

    end = False

    # run the loop until user quits.
    while not end:

        # pick random documents from test docs and provide titles to the user.
        user_docs = random.sample(test_docs, option_count)

        while True:
            print("\n---Available Choices For Articles(Titles)---\n")

            for i in range(len(user_docs)):
                print(str(i+1) + ": " + user_docs[i].title)

            print("r: Refresh List")
            print("q: Quit()\n")

            choice = raw_input("Enter Choice: ")

            if choice == 'q':
                end = True
                break
            elif choice == 'r':
                break
            else:
                try:
                    user_choice = int(choice) - 1
                    if user_choice < 0 or user_choice >= len(user_docs):
                        print("Invalid Choice.. Try Again..")
                        continue
                except:
                    print("Invalid Choice.. Try Again..")
                    continue
                selected_doc = user_docs[user_choice]

                # classifiers are sorted according to their f_measure in decreasing order. It helps when all
                # three classifiers differ in their predictions.
                classifier_list = sorted(classifier_list, key=lambda cl: cl.stats['f_measure'], reverse=True)

                prediction_list = list()
                for classifier in classifier_list:
                    prediction_list.append(classifier.classify([selected_doc])[0])

                prediction_count = Counter(prediction_list)
                top_prediction = prediction_count.most_common(1)

                if top_prediction[0][1] > 1:
                    prediction = top_prediction[0][0]
                else:
                    prediction = prediction_list[0]

                # create knn instance using documents of predicted topic. and find k closest documents.
                knn = KNN(all_docs[prediction])
                k_neighbours = knn.find_k_neighbours(selected_doc, k_n)

                while True:
                    print("\nRecommended Articles for : " + selected_doc.title)
                    for i in range(len(k_neighbours)):
                        print(str(i+1) + ": " + k_neighbours[i].title)
                    next_choice = raw_input("\nEnter Next Choice: [Article num to read the article. "
                                            "'o' to read the original article. "
                                            "'b' to go back to article choice list.]  ")

                    if next_choice == 'b':
                        break
                    elif next_choice == 'o':
                        text = selected_doc.text
                        print("\nArticle Text for original title : " + selected_doc.title)
                        print(text)
                    else:
                        try:
                            n_choice = int(next_choice) - 1
                            if n_choice < 0 or n_choice >= k_n:
                                print("Invalid Choice.. Try Again..")
                                continue
                        except:
                            print("Invalid Choice.. Try Again..")
                            continue
                        text = k_neighbours[n_choice].text
                        print("\nArticle Text for recommended title : " + k_neighbours[n_choice].title)
                        print(text)


def main():

    start_time = time.time()

    # Read documents, divide according to the topics and separate train and test data-set.

    t_path = "../dataset/bbc/"

    all_docs = defaultdict(lambda: list())

    topic_list = list()

    print("Reading all the documents...\n")

    for topic in os.listdir(t_path):
        d_path = t_path + topic + '/'
        topic_list.append(topic)
        temp_docs = list()

        for f in os.listdir(d_path):
            f_path = d_path + f
            temp_docs.append(Document(f_path, topic))

        all_docs[topic] = temp_docs[:]

    fold_count = 10

    train_docs, test_docs = list(), list()

    for key, value in all_docs.items():
        random.shuffle(value)
        test_len = int(len(value)/fold_count)
        train_docs += value[:-test_len]
        test_docs += value[-test_len:]

    # Create tfidf and tfidfie index of training docs, and store into the docs.
    index = Index(train_docs)

    print("Train Document Count: " + str(len(train_docs)))
    print("Test  Document Count: " + str(len(test_docs)))

    test_topics = [d.topic for d in test_docs]

    for doc in train_docs:
        doc.vector = doc.tfidfie

    for doc in test_docs:
        doc.vector = doc.tf

    # create classifier instances.
    nb = NaiveBayes()
    rc = RankClassifier()
    kmeans = KMeans(topic_list)
    
    classifier_list = [rc, nb, kmeans]

    for i in range(len(classifier_list)):

        print("\nClassifier #" + str(i+1) + "\n")

        classifier = classifier_list[i]

        classifier.confusion_matrix, c_dict = init_confusion_matrix(topic_list)

        print("Training...\n")

        classifier.train(train_docs)

        print("Testing... Classifying the test docs...\n")

        predictions = classifier.classify(test_docs)

        # Update the confusion matrix and statistics with updated values.
        classifier.confusion_matrix = update_confusion_matrix(test_topics, predictions, classifier.confusion_matrix,
                                                              c_dict)

        classifier.stats = cal_stats(classifier.confusion_matrix)

        print("Confusion Matrix\n")
        for item in classifier.confusion_matrix:
            print(item)

        print("\nStatistics\n")
        print_table(get_stats_table(classifier.stats))

    print("Run time...{} secs \n".format(round(time.time() - start_time, 4)))

    # call recommendation system once classifiers are ready.
    recommendation(all_docs, test_docs, classifier_list)


if __name__ == '__main__':
    main()
