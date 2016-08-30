'''
Created on Mar 30, 2016
@author: anup
'''

from collections import Counter
from collections import defaultdict
import math


class KMeans(object):

    def __init__(self,topics):
        self.topics = topics
        self.k = len(topics)
        self.confusion_matrix = None
        
    def prune_terms(self,docs, min_df=3):
        """
        Prune Terms which do not occur on min_df number of documents
        """
        term_doc_freq = Counter()
        for doc in docs:
            for term in doc.keys():
                term_doc_freq[term] += 1
                
        result = []
        for doc in docs:
            freq = Counter()
            for term in doc.keys():
                if term_doc_freq[term] >= min_df:
                    freq[term] += doc[term]
            if freq:
                result.append(freq)
        
        return result
    
    def doc_to_terms(self,docs):
        
        all_docs = defaultdict(lambda: list())
        
        for doc in docs:
            all_docs[doc.topic].append(doc.document_terms())
        
        return all_docs
    
    def train(self,docs):
        
        self.k_cluster = defaultdict(lambda: [])
        self.doc_norm = defaultdict(lambda: 0.0)
        
        documents = self.doc_to_terms(docs)
        
        for topic in self.topics:
            documents[topic] = self.prune_terms(documents[topic], 2)
        
        all_docs = []
        did = 0
        for topic in self.topics:
            for doc in documents[topic]:
                self.k_cluster[topic].append(did)
                all_docs.append(doc)
                self.doc_norm[did] = self.sqnorm(doc)
                did += 1
        
        self.documents = all_docs
        self.compute_means()
        
        for j in range(10):
            self.compute_clusters(self.documents)
            self.compute_means()            
            num_of_docs = []
            for i in self.k_cluster:
                num_of_docs.append(len(self.k_cluster[i]))
        
    def compute_means(self):

        self.mean_vectors = defaultdict(lambda: [])

        for topic in self.topics:
            term_freq = Counter()

            for doc_id in self.k_cluster[topic]:
                term_freq.update(self.documents[doc_id])

            if len(self.k_cluster[topic]) > 0:
                for term in term_freq:
                    term_freq[term] = 1.0 * term_freq[term] / len(self.k_cluster[topic])
                self.mean_vectors[topic] = term_freq
                
            self.mean_norms = defaultdict(lambda: 0.0)
            for t in self.mean_vectors.keys():
                self.mean_norms[t] = self.sqnorm(self.mean_vectors[t])
            
    def compute_clusters(self, documents):
        
        self.k_cluster = defaultdict(lambda: [])

        for doc_id in range(len(documents)):
            assign_cluster = -1
            min_distance = -1

            for cluster in self.topics:
                distance = self.distance(documents[doc_id],self.mean_vectors[cluster],self.mean_norms[cluster]+self.doc_norm[doc_id])
                if distance < min_distance or assign_cluster == -1:
                    assign_cluster = cluster
                    min_distance = distance

            self.k_cluster[assign_cluster].append(doc_id)
    
    def classify(self,test_docs):
        
        predictions = list()

        for doc in test_docs:
            (cluster, score) = self.assigned_cluster(doc.document_terms())
            predictions.append(cluster)
        
        return predictions
        
    def assigned_cluster(self,document):

        doc_norm = self.sqnorm(document)
        assigned_cluster = str()
        min_distance = -1

        for cluster in self.topics:
            distance = self.distance(document, self.mean_vectors[cluster], self.mean_norms[cluster]+doc_norm)

            if assigned_cluster == '' or distance < min_distance:
                min_distance = distance
                assigned_cluster = cluster
    
        return assigned_cluster, min_distance
    
    def sqnorm(self, d):

        sqsum = 0.0

        for key in d.keys():
            sqsum += (d[key]**2)
        
        return sqsum
    
    def distance(self, doc, mean, mean_norm):

        distance = mean_norm

        for term in doc:
            distance += - ( 2.0 * doc[term] * mean[term] )
            res = float(math.sqrt(distance))
            
        return res

    def error(self, documents):

        error = 0.0

        try:
            self.k_cluster_dist = defaultdict(lambda: [])

            for cluster in self.k_cluster.keys():

                for doc_id in self.k_cluster[cluster]:
                    distance = self.distance(documents[doc_id], self.mean_vectors[cluster], self.mean_norms[cluster]
                                             +self.doc_norm[doc_id])
                    error += distance
                    self.k_cluster_dist[cluster].append((documents[doc_id], distance))
        except IndexError:
            print('Error')

        return error

    def print_top_docs(self, n=10):

        for cluster in self.k_cluster.keys():
            print("CLUSTER ",cluster)
            topdocs = sorted(self.k_cluster_dist[cluster], key=lambda x:x[1])
            count = 0

            for doc_id in range(len(topdocs)):
                if len(topdocs[doc_id][0]) > 3:
                    buf = ' '.join(sorted(topdocs[doc_id][0].keys())).encode('utf-8')
                    print(buf.decode('utf-8'))
                    count += 1
                if count == n:
                    break