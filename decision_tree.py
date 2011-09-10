import numpy as np 
import scipy 
import scipy.weave 
import scipy.stats 
import random 
import math 

class ConstantLeaf:
    def __init__(self, v):
        self.v = v
    
    def predict(self, data):
        return self.v 

class TreeNode:
    def __init__(self, feature_idx, split_val, left, right):
        self.feature_idx = feature_idx
        self.split_val = split_val 
        self.left = left
        self.right = right
        
    def predict(self, data):
        x = data[self.feature_idx] 
        if x <= self.split_val:
            return self.left.predict(data)
        else:
            return self.right.predict(data) 
            
    def __str__(self):
        return str(self.__dict__)
        
def slow_gini(classes, labels):
    sum_squares = 0.0
    n = len(labels)
    n_squared = float(n * n)
    for c in classes:
        count = np.sum(labels == c)
        p_squared = count*count / n_squared
        sum_squares += p_squared
    return 1 - sum_squares 

def slow_eval_split(classes, left_labels, right_labels): 
    left_score = slow_gini(classes, left_labels)
    right_score = slow_gini(classes, right_labels)    
    n_left = len(left_labels)
    n_right = len(right_labels)
    n = float(n_left+n_right)
    
    combined_score = (n_left/n)*left_score + (n_right/n)*right_score 
    return combined_score 

inline = scipy.weave.inline 
def gini(classes, labels): 
    code = """
        int num_classes = Nclasses[0]; 
        int n = Nlabels[0];
        float sum_squares = 0.0f; 
        for (int class_index = 0; class_index < num_classes; ++class_index) { 
            int c = classes[class_index]; 
            int count = 0; 
            for (int i = 0; i < n; ++i) { 
                if (labels[i] == c) { ++count; } 
            }
            float p = ((float) count) / n; 
            sum_squares += p * p; 
        }
        return_val = 1.0f - sum_squares; 
    """
    return inline(code, ['classes', 'labels'], local_dict=None, verbose=2)

    
def eval_gini_split(classes, left_labels, right_labels): 
    code = """
        int num_classes = Nclasses[0]; 
        int nleft = Nleft_labels[0];
        int nright = Nright_labels[0]; 
        
        float left_sum_squares = 0.0f; 
        float right_sum_squares = 0.0f; 
        
        for (int class_index = 0; class_index < num_classes; ++class_index) { 
            int c = classes[class_index]; 
            int count = 0; 
            
            for (int i = 0; i < nleft; ++i) { 
                if (left_labels[i] == c) { ++count; } 
            }
            float p = ((float) count) / nleft; 
            left_sum_squares += p * p; 
            
            count = 0; 
            for (int i = 0; i < nright; ++i) { 
                if (right_labels[i] == c) { ++count; } 
            }
            p = ((float) count) / nright; 
            right_sum_squares += p*p; 
            
        }
        float left_gini = 1.0f - left_sum_squares; 
        float right_gini = 1.0f - right_sum_squares; 
        float total = (float) (nleft + nright);
        float left_weight = nleft / total; 
        float right_weight = nright / total; 
         
        return_val = left_weight * left_gini + right_weight  * right_gini; 
    """
    return inline(code, ['classes', 'left_labels', 'right_labels'], local_dict=None, verbose=2)
    
class RandomizedTree:
    def __init__(self, classes = None, num_features_per_node=None, min_leaf_size=5, max_height = 1000, thresholds='all'):
        self.root = None 
        self.num_features_per_node = num_features_per_node 
        self.min_leaf_size = min_leaf_size
        self.max_height = max_height 
        self.classes = None 
        self.nclasses = 0 
        if thresholds == 'all':
            self.get_thresholds = self.all_thresholds
        else:
            self.nthresholds = thresholds 
            self.get_thresholds = self.threshold_subset 
    def __str__(self):
        return str(self.root)
        
    def threshold_subset(self, x):
        unique_vals = np.unique(x)
        num_unique_vals = len(unique_vals)
        k = self.nthresholds
        if num_unique_vals <= k: return unique_vals
        else:
            step = max(num_unique_vals / (k+1), 1)
            return unique_vals[step::step]
    
    # get midpoints between all unique values         
    def all_thresholds(self, x): 
        unique_vals = np.unique(x)
        return (unique_vals[:-1] + unique_vals[1:]) / 2.0
            
    def majority(self, labels): 
        votes = np.zeros(self.nclasses)
        for i, c in enumerate(self.classes):
            votes[i] = np.sum(labels == c)
        majority_idx = np.argmax(votes)
        return self.classes[majority_idx]

    def split(self, data, labels, m, height):
        nfeatures = data.shape[1]
        # randomly draw m feature indices. 
        # should be more efficient than explicitly constructing a permutation
        # vector and then keeping only the head elements 
        random_feature_indices = random.sample(xrange(nfeatures), m)
        best_split_score = np.inf
        best_feature_idx = None
        best_thresh = None 
        best_left_indicator = None 
        best_right_indicator = None 
        classes = self.classes
        
        for feature_idx in random_feature_indices:
            feature_vec = data[:, feature_idx]
            thresholds = self.get_thresholds(feature_vec)
            for thresh in thresholds:
                left_indicator = feature_vec < thresh
                right_indicator = ~left_indicator
                
                left_labels = labels[left_indicator] 
                right_labels = labels[right_indicator] 
            
                combined_score = eval_gini_split(classes, left_labels, right_labels)
                if combined_score < best_split_score:
                    best_split_score = combined_score
                    best_feature_idx = feature_idx
                    best_thresh = thresh 
                    best_left_indicator = left_indicator
                    best_right_indicator = right_indicator 
    
        left_data = data[best_left_indicator, :] 
        left_labels = labels[best_left_indicator] 
        left_node = self.mk_node(left_data, left_labels, m, height+1)
        right_data = data[best_right_indicator, :] 
        right_labels = labels[best_right_indicator]
        right_node = self.mk_node (right_data, right_labels, m, height+1)
        node = TreeNode(best_feature_idx, best_thresh, left_node, right_node)
        return node 

    def mk_node(self, data, labels, m, height):
        # if labels are all same 
        if len(labels) <= self.min_leaf_size or height > self.max_height:
            self.nleaves += 1
            return ConstantLeaf(self.majority(labels))
            
        elif np.all(labels == labels[0]):
            self.nleaves += 1
            return ConstantLeaf(labels[0])
        else:
            return self.split(data, labels, m, height)
                
    def fit(self, data, labels): 
        if self.classes is None: 
            self.classes = np.unique(labels)
            self.nclasses = len(self.classes)
        self.nleaves = 0 
        nrows = data.shape[0]
        nfeatures = data.shape[1]
        if self.num_features_per_node is None:
            m = int(round(math.log(nfeatures, 2)))
        else:
            m = self.num_features_per_node 
        self.root = self.mk_node(data, labels, m, 1)

    def predict(self, vec):
        return self.root.predict(vec) 
