import numpy as np

class Member:
    def __init__(self,r_d,label=None,doc_id=None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id
    
class Cluster:
    def __init__(self) -> None:
        self._centroid = None
        self._members = []
    
    def resetMembers(self):
        self._members = []
    
    def addMembers(self,member):
        self._members.append(member)
    
from collections import defaultdict
import random
import os
class Kmeans():
    def __init__(self,num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)]
        self._E = [] # list of centroid
        self._S = 0 # overall simularity
        self._centroids = []
    def loadData(self,path):
        def sparseToDense(sparse_r_d, vocab_size):
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidf = sparse_r_d.split()
            for index_tfidf in indices_tfidf:
                index = int(index_tfidf.split(':')[0])
                tfidf = float(index_tfidf.split(':')[1])
                r_d[index] = tfidf
            return np.array(r_d)

        with open(os.path.join(path, "data_tf_idf.txt")) as f:
            d_lines = f.read().splitlines()
        with open(os.path.join(path, "words_idfs.txt")) as f:
            vocab_size = len(f.read().splitlines())

        self._data = []
        self._label_count = defaultdict(int)
        for data_id, d in enumerate(d_lines):
            features = d.split('<fff>')
            label, doc_id = int(features[0]),int(features[1])
            self._label_count[label]+=1
            r_d = sparseToDense(sparse_r_d=features[2],vocab_size=vocab_size)
            self._data.append(Member(r_d,label,doc_id))
    
    def randomInit(self,seed_value):
        random.seed(seed_value)
        data_shuffled = random.choices(self._data, k=self._num_clusters)
        for i in range(self._num_clusters):
            centroid = data_shuffled[i]._r_d
            self._clusters[i]._centroid = centroid
            self._centroids.append(centroid)
    
    def computeSimilarity(self,member,centroid):
        return sum(member._r_d * centroid) / np.sqrt(np.sum(member._r_d ** 2) * np.sum(centroid ** 2))
    
    def selectClusterFor(self,member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.computeSimilarity(member, cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        # add data point to cluster._members (list)
        best_fit_cluster.addMembers(member)
        return max_similarity

    def updateCentroidOf(self,cluster):
        average_rd = np.mean([member._r_d for member in cluster._members], axis=0)
        sum_squares = np.sum(average_rd ** 2)
        # normalize average rd
        new_centroid = np.array([num / np.sqrt(sum_squares) for num in average_rd])

        cluster._centroid = new_centroid
    
    def stoppingCondition(self,criterion,threshold):
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria

        if criterion == 'max_iters':
            if self._iteration  >= threshold:
                return True
            else:
                return False
        elif criterion == 'centroid':
            # after calculating centroids, before re-positioning centroid
            centroids_new = [list(cluster._centroid) for cluster in self._clusters]
            centroids_changes = [centroid for centroid in centroids_new if centroid not in self._centroids]
            self._centroids = centroids_new
            
            if len(centroids_changes) <= threshold:
                return True
            else:
                return False
        
        elif criterion == 'similarity':
            # while running algorithm, keep separated previous clustering error (S) and new clustering error (new_S)
            clustering_error_change = self._new_S - self._S
            self._S = self._new_S
            if clustering_error_change <= threshold:
                return True
            else:
                return False
    
    def run(self,seed_value,criterion,threshold):
        self.randomInit(seed_value)
        self._iteration = 0
        while True:
            for cluster in self._clusters:
                cluster.resetMembers()
            self._new_S = 0
            for member in self._data:
                max_s = self.selectClusterFor(member)
                self._new_S+=max_s
            for cluster in self._clusters:
                self.updateCentroidOf(cluster)

            self._iteration+=1
            if self.stoppingCondition(criterion,threshold):
                break

    
    def computePurity(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members ]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum+=max_count
            return majority_sum * 1. / len(self._data)
    
    def computeNMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1. 
            H_omega += - wk / N * np.log10(wk / N)
            member_labels = [member._label for member in cluster._members]
        for label in range(20):
            wk_cj = member_labels.count(label) * 1.
            cj = self._label_count[label]
            I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(20):
            cj = self._label_count[label] * 1.
            H_C += - cj / N * np.log10(cj / N)
        return I_value * 2. / (H_omega + H_C)

# Load the data
def load_data(path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_and_tfidfs = sparse_r_d.split()
        for index_and_tfidf in indices_and_tfidfs:
            index = int(index_and_tfidf.split(':')[0])
            tfidf = float(index_and_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)    
                
    with open(os.path.join(path, "data_tf_idf.txt")) as f:
            data_lines = f.read().splitlines()
    with open(os.path.join(path, "words_idfs.txt")) as f:
        vocab_size = len(f.read().splitlines())

    data, labels = [], []
    for data_id, d in enumerate(data_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        data.append(r_d)
        labels.append(label)
    return data, np.array(labels)