"""
Utils for ensemble methods.
Authors: Sara Mathieson + Micah Painter
Date: 22 February 2024
"""

from collections import OrderedDict
import sys
import optparse

# my files
from Partition import *

def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run decision tree method')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file')
    parser.add_option('-T', '--T', type='int', help='max depth (optional)')
    parser.add_option('-p', '--thresh', type='float', help='max depth (optional)')

    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename',]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts
    pass

def read_arff(filename):
    """Read arff file into Partition format."""
    arff_file = open(filename,'r')
    data = [] # list of Examples
    F = OrderedDict() # dictionary

    header = arff_file.readline()
    line = arff_file.readline().strip()

    # read the attributes
    while line != "@data":
        line = line.replace('{','').replace('}','').replace(',','')
        tokens = line.split()
        name = tokens[1][1:-1]
        features = tokens[2:]

        # label
        if name != "class":
            F[name] = features
        else:
            first = tokens[2]
        line = arff_file.readline().strip()

    # read the examples
    for line in arff_file:
        tokens = line.strip().split(",")
        X_dict = {}
        i = 0
        for key in F:
            val = tokens[i]
            X_dict[key] = val
            i += 1
        label = -1 if tokens[-1] == first else 1
        # set weight to None for now
        data.append(Example(X_dict,label,None))

    arff_file.close()

    # set weights on each example
    n = len(data)
    for i in range(n):
        data[i].set_weight(1/n)

    partition = Partition(data, F)
    return partition

def parse_args_roc():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run decision tree method')

    parser.add_option('-T', '--T', type='int', help='max depth (optional)')

    (opts, args) = parser.parse_args()

    mandatories = ['T']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts
    pass
