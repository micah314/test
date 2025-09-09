"""
Run ensemble methods to create ROC curves.
Author: Micah Painter
Date: 22 February 2024
"""

from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import DecisionStump
import math
from collections import OrderedDict
import Partition
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import roc_curve
import util

def main():
    '''
    takes in number of trees from the command line
    main function for making the roc curve
    runs the ensemble methods created in random_forest and ada_boost
    uses these proportions of trees that classify as 0 or 1 to make an roc curve
    '''

    # initializing stuff
    opts = util.parse_args_roc()
    train_partition = util.read_arff('data/mushroom_train.arff')
    test_partition = util.read_arff('data/mushroom_test.arff')
    T = opts.T
    print(opts.T)

    test_y_values = []
    for example in test_partition.data:
        test_y_values.append(example.label)




    # random forest training
    random_forest_stumps = []
    for i in range(T):
        # create a bootstrap training data (data)
        data = bootstrap_training_data(train_partition)

        # create a random sampling of sqrt(p) features (F)
        j = round(math.sqrt(len(train_partition.F)))
        F = random.sample(list(train_partition.F.items()), j)
        F = OrderedDict(F)

        # create a new partition with data and F
        new_partition = Partition.Partition(data, F)

        # create a decision stump from it
        random_forest_stump = DecisionStump.DecisionStump(new_partition)

        # add that decision stump to our ensemble of decision stumps
        random_forest_stumps.append(random_forest_stump)



    # adaboost training
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
        pass







    # ROC curves:
    thresholds = np.linspace(0, 1, num=20)


    # ROC curve for random forest:
    # for each threshold, find the fp's and tp's for the training data
    x_data_rf = [] # true positives
    y_data_rf = [] # false positives

    for threshold in thresholds:
        pred_results = []
        for example in test_partition.data:
            num_pos = 0
            num_neg = 0

            # classify the example for each stump in our ensemble, keeping track of number of positve and negative classifications
            for stump in random_forest_stumps:
                classify = stump.classify(example.features, threshold)
                if classify == 1:
                    num_pos += 1
                else:
                    num_neg += 1
            
            # pick the majority label and add to predicted results
            if num_pos > num_neg:
                pred_results.append(1)
            else:
                pred_results.append(-1)

        # make a list of the true results
        true_results = []
        for example in test_partition.data:
            true_results.append(example.label)

        # make the confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_results, pred_results).ravel()
        y_data_rf.append(tp / (tp + fn))
        x_data_rf.append(fp / (fp + tn))
    

    # plot the false postives and true positives to get the ROC curve
    plt.plot(x_data_rf, y_data_rf, "-r", label = 'random forest')






    # ROC curve for adaboost
    y_data_ab = []
    x_data_ab = []

    for threshold in thresholds:
        pred_results = []
        for example in test_partition.data:
            total_value = 0
            for stump in ensemble:
                total_value += ensemble[stump] * stump.classify(example.features, threshold) # ensemble[stump] is alpha, which is how much we 'trust' the classifier
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
        y_data_ab.append(tp / (tp + fn))
        x_data_ab.append(fp / (fp + tn))


    plt.plot(x_data_ab, y_data_ab, "-b", label = 'adaboost')

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC Curve for Mushroom Dataset, T=100")
    plt.legend()
    plt.savefig("figs/rf_vs_ada_T100.png")
    plt.show()

    print(auc(x_data_rf, y_data_rf))
    print(auc(x_data_ab, y_data_ab))






###### Helper funcitons ######
    
def plot_ROC (x_axis, y_axis, legend_name):
    '''
    plots a curve (in this case ROC) given x data, y data, and a name for the data
    does not return anything
    '''

    # generate a random color
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)

    # set all the labels
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("Zip code predictions")

    # plot it with legend
    plt.plot(x_axis, y_axis, 'o-', color = color, label = legend_name)
    plt.legend()
    
def bootstrap_training_data(train_partition):
    '''
    bootstrapping function for the training data
    takes in a training partition and returns a partition of n random pieces of the data (with repetition)
    '''
    # create a list of all possible examples (partition.data is a list)
    new_data = []
    old_data = train_partition.data
    for i in range(train_partition.n):
        new_data.append(random.choice(old_data))
    return new_data
    # for n examples where n is the size of the dataset
    # randomly select one example and add it to our new list

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
