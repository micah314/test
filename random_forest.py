"""
Implements Random Forests with decision stumps.
Author: Micah Painter
Date: 22 February 2024
"""

from sklearn.metrics import confusion_matrix
import DecisionStump
import math
from collections import OrderedDict
import Partition
import random
import util

def main():
    '''
    Main function for creating random forests with decision stumps.
    Reads in the data from the command line 
    Creates a random forest finds a confusion matrix of testing data from the results
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
    
    print(train_partition.n)
    print(test_partition.n)
    print(T, ", ", thresh)


    # Uncomment this to see weighted example
    # train_stump = DecisionStump.DecisionStump(train_partition)
    # for i in range(train_partition.n):
    #     example = train_partition.data[i]
    #     if i == 0 or i == 8:
    #         example.set_weight(0.25)
    #     else:
    #         example.set_weight(0.5/(train_partition.n-2))
    #     train_partition.data[i] = example
    # train_stump_2 = DecisionStump.DecisionStump(train_partition)
    # for example in test_partition.data:
    #     print(train_stump_2.classify(example.features, thresh))
    # print(train_stump_2.name, train_stump_2.children)





    # Creating a random forest
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

    # find the predicted results
    pred_results = []

    # for each example in the testing data, classify the num of positives and negatives and "vote" on the values
    for example in test_partition.data:
        num_pos = 0
        num_neg = 0

        # classify the example for each stump in our ensemble, keeping track of number of positve and negative classifications
        for stump in random_forest_stumps:
            classify = stump.classify(example.features, thresh)
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
    print(confusion_matrix(true_results, pred_results))
    print("false positive rate: ", fp / (fp + tn))
    print("true positive rate:", tp / (tp + fn))

    
    

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


if __name__ == "__main__":
    main()
