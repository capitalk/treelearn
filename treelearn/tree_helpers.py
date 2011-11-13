# TreeLearn
#
# Copyright (C) Capital K Partners
# Author: Alex Rubinsteyn
# Contact: alex [at] capitalkpartners [dot] com 
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.


import numpy as np 
import scipy
import scipy.weave 


# some sklearn classifiers leave behind large data members after fitting
# which make serialization a pain--- clear those fields 
def clear_sklearn_fields(clf):
    if hasattr(clf, 'label_'):
        clf.label_ = None
    if hasattr(clf, 'sample_weight'):
        clf.sample_weight = None 

def midpoints(x):
    return (x[1:] + x[:-1])/2.0
    
def majority(labels, classes=None): 
    if classes is None: 
        classes = np.unique(labels)
    votes = np.zeros(len(classes))
    for i, c in enumerate(classes):
        votes[i] = np.sum(labels == c)
    majority_idx = np.argmax(votes)
    return classes[majority_idx] 


def slow_gini(classes, labels):
    sum_squares = 0.0
    n = len(labels)
    if n == 0:
        return 0.0
    else:
        n_squared = float(n * n)
        for c in classes:
            count = np.sum(labels == c)
            p_squared = count*count / n_squared
            sum_squares += p_squared
        return 1 - sum_squares 

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


def slow_eval_split(classes, feature_vec, thresh, labels): 
    left_labels = labels[feature_vec <= thresh]
    right_labels = labels[feature_vec <= thresh] 
    left_score = slow_gini(classes, left_labels)
    right_score = slow_gini(classes, right_labels)    
    n_left = len(left_labels)
    n_right = len(right_labels)
    n = float(n_left+n_right)
    
    combined_score = (n_left/n)*left_score + (n_right/n)*right_score 
    return combined_score 

    
dtype2ctype = {
    np.dtype(np.float64): 'double',
    np.dtype(np.float32): 'float',
    np.dtype(np.int32): 'int',
    np.dtype(np.int16): 'short',
    np.dtype(np.bool): 'bool', 
}
   
def eval_gini_split(classes, feature_vec, thresh, labels): 
    left_mask = feature_vec <= thresh
    code = """
        
        int num_classes = Nclasses[0]; 
        int nlabels = Nlabels[0]; 
        
        float left_sum_squares = 0.0f; 
        float right_sum_squares = 0.0f; 
        
        
        /* total number of elements in the left and right of the split */ 
        int total_left = 0; 
        int total_right = 0; 
        
        /* first pass for C = 0 to get total counts along with class-specific
           scores 
        */
        int left_class_count = 0; 
        int right_class_count = 0; 
        
        for (int i = 0; i < nlabels; ++i) { 
            if (left_mask[i]) {
                total_left += 1; 
                if (labels[i] == 0) left_class_count += 1; 
            } else {
                total_right += 1;
                if (labels[i] == 0) right_class_count += 1; 
            }
        }
        if (total_left > 0) {
            float left_p = ((float) left_class_count) / total_left; 
            left_sum_squares += left_p * left_p; 
        }
        if (total_right > 0) { 
            float right_p = ((float) right_class_count) / total_right; 
            right_sum_squares += right_p* right_p; 
        }
        
        /* how many elements of each side have we counted in the score so far? */ 
        int cumulative_left_count = left_class_count; 
        int cumulative_right_count = right_class_count; 
        
        /* if we have a multi-class problem iterate over rest of classes, 
           except for the last class, whose size can be inferred from the 
           difference between left_count and total_left
        */ 
        for (int class_index = 1; class_index < num_classes - 1; ++class_index) { 
            int c = classes[class_index]; 
            left_class_count = 0; 
            right_class_count = 0; 
            
            for (int i = 0; i < nlabels; ++i) {
                if (labels[i] == c) { 
                    if (left_mask[i]) left_class_count += 1; 
                    else right_class_count += 1; 
                }
            }
            cumulative_left_count += left_class_count; 
            cumulative_right_count += right_class_count; 
            
            if (total_left > 0) {
                float left_p = ((float) left_class_count) / total_left; 
                left_sum_squares += left_p * left_p; 
            }
            if (total_right > 0) { 
                float right_p = ((float) right_class_count) / total_right; 
                right_sum_squares += right_p* right_p; 
            }
        }
        
        /* handle last class */ 
        left_class_count = total_left - cumulative_left_count; 
        right_class_count = total_right - cumulative_right_count; 
        if (total_left > 0) {
            float left_p = ((float) left_class_count) / total_left; 
            left_sum_squares += left_p * left_p; 
        }
        if (total_right > 0) { 
            float right_p = ((float) right_class_count) / total_right; 
            right_sum_squares += right_p* right_p; 
        }
        float left_gini = 1.0f - left_sum_squares; 
        float right_gini = 1.0f - right_sum_squares; 
        float total = (float) nlabels; 
        float left_weight = total_left / total; 
        float right_weight = total_right / total; 
         
        return_val = left_weight * left_gini + right_weight  * right_gini; 
    """ 
    return inline(code, ['classes', 'left_mask', 'labels'], \
        local_dict=None, verbose=2)


def slow_find_best_gini_split(classes, feature_vec, thresholds, labels): 
    best_t = None
    best_score = np.inf
    
    n = len(labels) 
    for t in thresholds:
        mask = feature_vec <= t
        left_labels = labels[mask]
        right_labels = labels[~mask] 
        left_score = slow_gini(classes, left_labels)
        right_score = slow_gini(classes, right_labels)    
        n_left = len(left_labels)
        n_right = len(right_labels)
        nf = float(n)
        combined_score = (n_left/nf)*left_score + (n_right/nf)*right_score 
        if combined_score < best_score:
            best_t = t
            best_score = combined_score
    return best_t, best_score 

def find_min_variance_split(feature_vec, thresholds, ys): 
    best_score = np.inf
    best_t = None
    for t in thresholds:
        mask = feature_vec <= t
        left = ys[mask]
        right = ys[~mask]
        left_size = left.shape[0]
        right_size = right.shape[0]
        
        if left_size > 0 and right_size > 0:
            total = float(left_size + right_size)
            score = (left_size / total) * np.var(left) + (right_size / total) * np.var(right)
            if score < best_score:
                best_score = score
                best_t = t
    return best_t, best_score 
    
def find_best_gini_split(classes, feature_vec, thresholds, labels): 
    code = """
        
        int num_classes = Nclasses[0]; 
        int n_labels = Nlabels[0]; 
        int n_thresholds = Nthresholds[0]; 
        
        float best_score = 10000000.0; 
        double best_thresh = 0.0; 
        
        /* loop over each possible threshold, compute GINI impurity for each,
           return the threshold with lowest score 
        */ 
        for (int t_index = 0; t_index < n_thresholds; t_index++) {
            double thresh = thresholds[t_index];
            
            float left_sum_squares = 0.0f; 
            float right_sum_squares = 0.0f; 
        
        
            /* total number of elements in the left and right of the split */ 
            int total_left = 0; 
            int total_right = 0; 
        
            /* first pass for C = 0 to get total counts along with class-specific
               scores 
            */
            int left_class_count = 0; 
            int right_class_count = 0; 
            
            for (int i = 0; i < n_labels; ++i) { 
                if (feature_vec[i] <= thresh) {
                    total_left += 1; 
                    if (labels[i] == 0) left_class_count += 1; 
                } else {
                    total_right += 1;
                    if (labels[i] == 0) right_class_count += 1; 
                }
            }
            if (total_left > 0) {
                float left_p = ((float) left_class_count) / total_left; 
                left_sum_squares += left_p * left_p; 
            }
            if (total_right > 0) { 
                float right_p = ((float) right_class_count) / total_right; 
                right_sum_squares += right_p* right_p; 
            }
            
            /* how many elements of each side have we counted in the score so far? */ 
            int cumulative_left_count = left_class_count; 
            int cumulative_right_count = right_class_count; 
            
            /* if we have a multi-class problem iterate over rest of classes, 
               except for the last class, whose size can be inferred from the 
               difference between left_count and total_left
            */ 
            for (int class_index = 1; class_index < num_classes - 1; ++class_index) { 
                int c = classes[class_index]; 
                left_class_count = 0; 
                right_class_count = 0; 
                
                for (int i = 0; i < n_labels; ++i) {
                    if (labels[i] == c) { 
                        if (feature_vec[i] <= thresh) left_class_count += 1; 
                        else right_class_count += 1; 
                    }
                }
                cumulative_left_count += left_class_count; 
                cumulative_right_count += right_class_count; 
                
                if (total_left > 0) {
                    float left_p = ((float) left_class_count) / total_left; 
                    left_sum_squares += left_p * left_p; 
                }
                if (total_right > 0) { 
                    float right_p = ((float) right_class_count) / total_right; 
                    right_sum_squares += right_p* right_p; 
                }
            }
            
            /* handle last class */ 
            left_class_count = total_left - cumulative_left_count; 
            right_class_count = total_right - cumulative_right_count; 
            if (total_left > 0) {
                float left_p = ((float) left_class_count) / total_left; 
                left_sum_squares += left_p * left_p; 
            }
            if (total_right > 0) { 
                float right_p = ((float) right_class_count) / total_right; 
                right_sum_squares += right_p* right_p; 
            }
            float left_gini = 1.0f - left_sum_squares; 
            float right_gini = 1.0f - right_sum_squares; 
            float total = (float) n_labels; 
            float left_weight = total_left / total; 
            float right_weight = total_right / total; 
            float score = left_weight * left_gini + right_weight  * right_gini; 
            if (score < best_score) { 
                best_score = score; 
                best_thresh = thresh; 
            }
        }
        py::tuple results(2);
        results[0] = best_thresh;
        results[1] = best_score;
        return_val = results;
    """ 
    return inline(code, ['classes', 'feature_vec', 'thresholds', 'labels'], \
        local_dict=None, verbose=2)
