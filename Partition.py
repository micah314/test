"""
Example and Partition data structures.
Authors: Sara Mathieson + Micah Painter
Date: 22 February 2024
"""
import math

class Example:

    def __init__(self, features, label, weight):
        """
        Class to hold an individual example (data point) and its weight.
        features -- dictionary of features for this example (key: feature name,
            value: feature value for this example)
        label -- label of the example (-1 or 1)
        weight -- weight of the example (starts out as 1/n)
        """
        self.features = features
        self.label = label
        self.weight = weight

    def set_weight(self, new):
        """Change the weight on an example (used for AdaBoost)."""
        self.weight = new

class Partition:

    def __init__(self, data, F):
        """
        Class to hold a set of examples and their associated features, as well
        as compute information about this partition (i.e. entropy).
        data -- list of Examples
        F -- dictionary (key: feature name, value: list of feature values)
        """
        self.data = data
        self.F = F
        self.n = len(self.data)

    def _cond_prob(self, f, val, c):
        """Compute P(Y=c|X=val)."""
        y_count = 0
        denom = 0
        for i in range(self.n):
            example = self.data[i]
            if example.features[f] == val:
                denom += self.data[i].weight # here
                if example.label == c:
                    y_count += self.data[i].weight # here
        # case when we don't have any examples with this feature value
        if denom == 0:
            return 0
        return y_count/denom

    def _prob(self, c):
        """Compute P(Y=c)."""
        denom = 0
        y_count = 0
        for i in range(self.n):
            example = self.data[i]
            if example.label == c:
                y_count += example.weight # here
            denom += example.weight # here
        return y_count/denom

    def _entropy(self):
        """Compute H(Y)."""
        entropy = 0
        # consider only labels -1 and +1
        for c in [-1,1]:
            p = self._prob(c)
            if p != 0:
                entropy -= p*math.log(p,2) # log base 2
        return entropy

    def _cond_entropy(self, feature, val):
        """Compute H(Y|X=val)."""
        entropy = 0
        # consider only labels -1 and +1
        for c in [-1,1]:
            p = self._cond_prob(feature,val,c)
            if p != 0:
                entropy -= p*math.log(p,2) # log base 2
        return entropy

    def _probX(self, f, val):
        """Compute P(X=val)."""
        val_count = 0
        denom = 0
        for i in range(self.n):
            example = self.data[i]
            if example.features[f] == val:
                val_count += example.weight # here
            denom += example.weight # here
        return val_count/denom

    def _full_cond_entropy(self, feature):
        """Compute H(Y|X). Weighted average."""
        s = 0
        for val in self.F[feature]:
            s += self._probX(feature,val) * self._cond_entropy(feature,val)
        return s

    def _info_gain(self, feature):
        """Compute the information gain."""
        Hy = self._entropy() # H(Y)
        Hyx = self._full_cond_entropy(feature) # H(Y|X)
        return Hy-Hyx
    
    def best_feature(self):
        '''Return the feature with the highest information gain. Returns None if all examples have the same value'''
        highest_feature = None
        highest_info_gain = 0
        for feature in self.F:
            if self._info_gain(feature) > highest_info_gain:
                highest_info_gain = self._info_gain(feature)
                highest_feature = feature

        return highest_feature
    

    def prob_pos(self):
        '''Returns a dictionary where the keys are the features of the best value and the values are the probability of that feature value being postive'''
        best_feature = self.best_feature()
        prob_dict = {}

        # for each feature value, find the probability it's positive and store it in the dictionary
        for feature in self.F[best_feature]:
            num_positive = 0
            total_num = 0

            # for each example, update total probability and total number of examples
            for example in self.data:
                if example.features[best_feature] == feature:
                    if example.label == 1:
                        num_positive += example.weight # here we change 1 to the weight of the example 
                    total_num += example.weight # here too
            
            # update the dictionary where the key is the feautre and the value is the probability of a positive result
            if total_num == 0:
                prob_dict[feature] = 0
            else:
                prob_dict[feature] = num_positive/total_num
        
        return prob_dict


