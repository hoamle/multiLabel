import theano
import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, SliceLayer, ConcatLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
import lasagne

import time, sys
from natsort import natsorted
import numpy as np
import cPickle as pickle

from data_prep import data_prep
from nolearn.lasagne import NeuralNet, BatchIterator

# for passing training hyper-params when for eg learning rate decay
def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

# model definition
def build_model(num_epochs, batchsize):
    net = NeuralNet(
        layers=[
            ('input', InputLayer),
            ('conv1', ConvLayer),
            ('pool1', PoolLayer),
            ('conv2', ConvLayer),
            ('pool2', PoolLayer),
            ('conv3', ConvLayer),
            ('conv4', ConvLayer),
            ('conv5', ConvLayer),
            ('pool5', PoolLayer),
            ('hidden6', DenseLayer),
            ('dropout6', DropoutLayer),
            ('hidden7', DenseLayer),
            ('dropout7', DropoutLayer),
            ('output', DenseLayer)
            ],
        # layer params
        input_shape = (None, 3, 227, 227),
        conv1_num_filters=96, conv1_filter_size=(11,11), conv1_filter_stride=(4,4),
        pool1_pool_size=(3,3), pool1_pool_stride=(2,2),
        conv2_num_filters=256, conv2_filter_size=(5,5), conv2_filter_pad=(2,2),
        pool2_pool_size=(3,3), pool2_pool_stride=(2,2),
        conv3_num_filters=384, conv3_filter_size=(3,3), conv3_filter_pad=(1,1),
        conv4_num_filters=384, conv4_filter_size=(3,3), conv4_filter_pad=(1,1),
        conv5_num_filters=256, conv5_filter_size=(3,3), conv5_filter_pad=(1,1),
        pool5_pool_size=(3,3), pool5_pool_stride=(2,2),
        hidden6_num_units=4096, dropout6_p=0.5,
        hidden7_num_units=4096, dropout7_p=0.5,
        output_num_units=22, output_nonlinearity=lasagne.nonlinearities.softmax,

        # optimisation 
        objective=lasagne.objectives.categorical_crossentropy,
        # optimisation method
        update=lasagne.updates.nesterov_momentum,
        update_learning_rate=theano.shared(float32(0.01)),
        update_momentum=theano.shared(float32(0.9)),

        regression = False,
        #eval_size=0.3,
        batch_iterator_train=BatchIterator(batch_size=batchsize),
        max_epochs=num_epochs,
        verbose=1
    )
    return net
   
def main(num_epochs=500, mode="run", batchsize=96): 
    X, Y, imgMean_vals = data_prep(mode=mode)
    net0 = build_model(num_epochs, batchsize)
    net0.fit(X,Y)

if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['num_epochs'] = int(sys.argv[1])
    if len(sys.argv) > 2:
        kwargs['mode'] = sys.argv[2]
    
    main(**kwargs)