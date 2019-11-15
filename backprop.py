#!/usr/bin/env python3
import numpy as np
from io import StringIO

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
#DATA_PATH = "/u/cs246/data/adult/" #TODO: if doing development somewhere other than the cycle server, change this to the directory where a7a.train, a7a.dev, and a7a.test are

DATA_PATH = "/Users/shreevandanakachroo/Desktop/ML/MLHW2/adult"

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y,0) #treat -1 as 0 insteadbecause sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray([ys],dtype=np.float32).T, np.asarray(xs,dtype=np.float32) #returns a tuple, first is an array of labels, second is an array of feature vectors

def init_model(args):
    #w1 = None
    #w2 = None

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1,len(w2))
    else:
        #TODO (optional): If you want, you can experiment with a different random initialization. As-is, each weight is uniformly sampled from [-0.5,0.5).
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES) #bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1) #add bias column

    #At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.


    #TODO: Replace this with whatever you want to use to represent the network; you could use use a tuple of (w1,w2), make a class, etc.
    model = {}
    model['weight1'] = w1
    model['weight2'] = w2 
    #raise NotImplementedError #TODO: delete this once you implement this function
    return model


def activation(x):
    return 1.0/(1.0 + np.exp(-x))

def calculate_error(x,y):
	return np.square(x-y)

def sigmoid_derivative(x):
	return x * (1.0 - x)

def forward_propagation(model, train_ys, train_xs, args):
    weights_hidden = model['weight1'] 
    weights_output = model['weight2'] 
    a1 = np.dot(weights_hidden,train_xs)
    z1 = activation(a1)
    bias = np.ones((1, z1.shape[1]))
    z1_extended = (np.concatenate((z1,bias), axis = 0))
    z1_derivate = sigmoid_derivative(z1)
    a2 = np.dot(weights_output,z1_extended)
    z2 = activation(a2)
    z2_derivate = sigmoid_derivative(z2)
    return z1,z1_extended,z2,z1_derivate,z2_derivate

def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):
    #TODO: Implement training for the given model, respecting args

    weights_hidden = model['weight1'] 
    weights_output = model['weight2'] 
    hidden_size = weights_output.shape[1]
    for j in range(0,args.iterations):
        for i in range(0,len(train_xs)):
            train = train_xs[i].reshape(train_xs[i].shape[0],1)
            z1,z1_extended,z2,z1_derivate,z2_derivate = forward_propagation(model,train_ys[i],train,args)
            #error = calculate_error(z2,train_ys[i])
            weights_output_reduced = weights_output[:,0:hidden_size - 1]
            delta2 = (z2-train_ys[i])*z2_derivate
            delta1 = (z2-train_ys[i])*z2_derivate*(np.transpose(weights_output_reduced))*(z1_derivate)
            error_derivate_w2 = np.dot(delta2,np.transpose(z1_extended))
            error_derivate_w1 = np.dot(delta1,np.transpose(train))
            weights_output = weights_output - ((args.lr)*error_derivate_w2)
            weights_hidden = weights_hidden - ((args.lr)*error_derivate_w1)
            model['weight1'] = weights_hidden 
            model['weight2'] = weights_output
#raise NotImplementedError #TODO: delete this once you implement this function
    return model

def test_accuracy(model, test_ys, test_xs, args):
    accuracy = 0.0
    correct = 0.0
    weights_hidden = model['weight1']
    weights_output = model['weight2']
    #TODO: implement accuracy computation of given weight vector on the test data (i.e. how many test data points are classified correctly by the weight vector)
    for i in range(0,len(test_ys)):
        test = test_xs[i].reshape(test_xs[i].shape[0],1)
        z1,z1_extended,z2,z1_derivate,z2_derivate = forward_propagation(model,test_ys[i],test,args)
        if z2 <= 0.5:
            if test_ys[i]<=0:
                correct += 1
        elif z2 > 0.5:
            if test_ys[i]>0:
                correct += 1
    accuracy = correct /(len(test_ys))
    #TODO: Implement accuracy computation of given model on the test data
    #raise NotImplementedError #TODO: delete this once you implement this function

    return accuracy

def extract_weights(model):
    w1 = None
    w2 = None
    #TODO: Extract the two weight matrices from the model and return them (they should be the same type and shape as they were in init_model, but now they have been updated during training)
    w1 = model['weight1'] 
    w2 = model['weight2'] 
    #raise NotImplementedError #TODO: delete this once you implement this function
    return w1, w2

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1','W2'), type=str, help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')


    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)

    model = init_model(args)
    model1 = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)
    #print(model)
    accuracy = test_accuracy(model1, test_ys, test_xs, args)
    print('Test accuracy: {}'.format(accuracy))
    if not args.nodev:
        dev_accuracy = test_accuracy(model1, dev_ys, dev_xs, args)
        print('Dev accuracy: {}'.format(dev_accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1,w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2,w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))

if __name__ == '__main__':
    main()
