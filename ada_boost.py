"""
Implements AdaBoost algorithm with decision stumps.
Author: Micah Painter
Date: 22 February 2024
"""

from sklearn.metrics import confusion_matrix
import DecisionStump
import numpy as np
from collections import OrderedDict
import util

def main():
    '''
    main function for using the adaboost algorithm with decision stumps
    takes in arguments from the command line as the data files
    trains an ensemble of decision stumps, then uses that ensemble to test on the testing data
    prints a confusion matrix of the results
    '''

    # read in data (y in {-1,1})
    opts = util.parse_args()
    train_partition = util.read_arff(opts.train_filename)
    test_partition  = util.read_arff(opts.test_filename)
    T = opts.T
    thresh = opts.thresh
    if T == None:
        T = 10
    if thresh == None:
        thresh = 0.5


    # training:
    ensemble = OrderedDict()
    for t in range(T):
        # make a decision stump on the training data (no bootstrap)
        new_stump = DecisionStump.DecisionStump(train_partition)

        # use weighted_error function to compute weighted error for that decision stump (inputing all training points)
        weighted_error_value = weighted_error(train_partition.data, new_stump)

        # find alpha using the function classifier_score (input weighted error)
        alpha = classifier_score(weighted_error_value)

        # add an entry to our ordered dictionary ensemble with the key being the decision stump and the value being alpha
        ensemble[new_stump] = alpha

        # find ct to input into update_weights
        ct = compute_ct(train_partition.data, new_stump, alpha)

        # use update_weights function to update all the weights (input all training points and decision stump, alpha, ct)
        update_weights(train_partition.data, new_stump, alpha, ct)

    
    # testing:
    pred_results = []
    for example in test_partition.data:
        total_value = 0
        for stump in ensemble:
            total_value += ensemble[stump] * stump.classify(example.features, thresh) # ensemble[stump] is alpha, which is how much we 'trust' the classifier
        if total_value >= 0:
            pred_results.append(1)
        else:
            pred_results.append(-1)
    
    # make a list of the true results
    true_results = []
    for example in test_partition.data:
        true_results.append(example.label)

    # make the confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_results, pred_results).ravel()
    print(confusion_matrix(true_results, pred_results))
    print("false positive rate: ", fp / (fp + tn))
    print("true positive rate:", tp / (tp + fn))



def weighted_error(training_points, decision_stump):
    '''
    takes in training points (training_partition.data) and a single decision stump
    for each point, classify it based on the decision stump
        if that point's classification does not equal the true value, add it's weight to the total weighted error
    returns the weighted error for that decision stump
    '''
    weighted_error = 0
    for point in training_points:
        predicted_value = decision_stump.classify(point.features)
        true_value = point.label
        if not predicted_value == true_value:
            weighted_error += point.weight
        
    return weighted_error

def classifier_score(weighted_error):
    '''
    takes in the weighted error and uses that to compute alpha, the classifier score
    returns alpha
    '''
    score = 0.5 * np.log((1-weighted_error)/weighted_error)
    return score

def update_weights(training_points, decision_stump, alpha, ct):
    '''
    update the weights on each of the training points
    '''
    for point in training_points:
        predicted_value = decision_stump.classify(point.features)
        true_value = point.label
        new_weight = ct * point.weight * np.exp(-true_value * alpha * predicted_value)
        point.set_weight(new_weight)


def compute_ct(training_points, decision_stump, alpha):
    '''
    takes in a set of training points, a single decision stump, and an alpha value
    used to compute ct to normalize the data correctly in update_weights
    returns ct
    '''
    denom = 0
    for point in training_points:
        predicted_value = decision_stump.classify(point.features)
        true_value = point.label
        to_add = point.weight * np.exp(-true_value * alpha * predicted_value)
        denom += to_add

    if denom == 0:
        return 0
    return 1/denom

if __name__ == "__main__":
    main()
