"""
Decision stump data structure (i.e. tree with depth=1), non-recursive.
Authors: Sara Mathieson + Micah Painter
Date: 22 February 2024
"""

class DecisionStump:

    def __init__(self, partition):
        """
        Create a DecisionStump from the given partition of data by choosing one
        best feature and splitting the data into leaves based on this feature.
        """

        # find the best feature
        self.name = partition.best_feature()

        # create the dictionary
        self.children = partition.prob_pos()


        pass

    def get_name(self):
        """Getter for the name of the best feature (root)."""
        return self.name
    
    def get_children(self):
        '''Getter for the children of the best feature'''
        return self.children

    def add_child(self, edge, prob):
        """
        Add a child with edge as the feature value, and prob as the probability
        of a positive result.
        """
        self.children[edge] = prob

    def get_child(self, edge):
        """Return the probability of a positive result, given feature value."""
        return self.children[edge]

    def __str__(self):
        """Returns a string representation of the decision stump."""
        s = self.name + " =\n"
        for v in self.children:
            s += "  " + v + ", " + str(self.children[v]) + "\n"
        return s

    def classify(self, test_features, thresh=0.5):
        """
        Classify the test example (using features only) as +1 (positive) or -1
        (negative), using the provided threshold.
        returns 1 or -1, the predicted value
        """
        # find the feature value of the testing data that corresponds the best feature of the decision stump
        test_feature_value = test_features[self.name]

        # figure out if we should classify as +1 or -1 based on the dictionary of features and their corresponding probabilities
        if self.children[test_feature_value] >= thresh:
            return 1
        else:
            return -1
