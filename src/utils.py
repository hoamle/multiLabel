from lasagne.layers import InputLayer, LocalResponseNormalization2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer, DropoutLayer
import lasagne.nonlinearities

from natsort import natsorted
import cPickle as pickle
import numpy as np
import sklearn

def build_model(num_classes, pretrained_w_path="", input_var=None, drop_p=0.5):
    net = {}
    net['data'] = InputLayer(shape=(None, 3, 227, 227), input_var=input_var)

    # conv1
    net['conv1'] = Conv2DLayer(
        net['data'],
        num_filters=96,
        filter_size=(11, 11),
        stride = 4,
        nonlinearity=lasagne.nonlinearities.rectify)

    
    # pool1
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2)

    # norm1
    net['norm1'] = LocalResponseNormalization2DLayer(net['pool1'],
                                                     n=5,
                                                     alpha=0.0001/5.0,
                                                     beta = 0.75,
                                                     k=1)

    # conv2
    # The caffe reference model uses a parameter called group.
    # This parameter splits input to the convolutional layer.
    # The first half of the filters operate on the first half
    # of the input from the previous layer. Similarly, the
    # second half operate on the second half of the input.
    #
    # Lasagne does not have this group parameter, but we can
    # do it ourselves.
    #
    # see https://github.com/BVLC/caffe/issues/778
    # also see https://code.google.com/p/cuda-convnet/wiki/LayerParams
    
    # before conv2 split the data
    # norm1 size: 23 x 23 x 96
    net['conv2_data1'] = SliceLayer(net['norm1'], indices=slice(0, 48), axis=1)
    net['conv2_data2'] = SliceLayer(net['norm1'], indices=slice(48,96), axis=1)

    # now do the convolutions
    net['conv2_part1'] = Conv2DLayer(net['conv2_data1'],
                                     num_filters=128,
                                     filter_size=(5, 5),
                                     pad = 2)
    net['conv2_part2'] = Conv2DLayer(net['conv2_data2'],
                                     num_filters=128,
                                     filter_size=(5,5),
                                     pad = 2)

    # now combine
    net['conv2'] = concat((net['conv2_part1'],net['conv2_part2']),axis=1)
    
    # pool2
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride = 2)
    
    # norm2
    net['norm2'] = LocalResponseNormalization2DLayer(net['pool2'],
                                                     n=5,
                                                     alpha=0.0001/5.0,
                                                     beta = 0.75,
                                                     k=1)
    
    # conv3
    # no group
    net['conv3'] = Conv2DLayer(net['norm2'],
                               num_filters=384,
                               filter_size=(3, 3),
                               pad = 1)

    # conv4
    # group = 2
    net['conv4_data1'] = SliceLayer(net['conv3'], indices=slice(0, 192), axis=1)
    net['conv4_data2'] = SliceLayer(net['conv3'], indices=slice(192,384), axis=1)
    net['conv4_part1'] = Conv2DLayer(net['conv4_data1'],
                                     num_filters=192,
                                     filter_size=(3, 3),
                                     pad = 1)
    net['conv4_part2'] = Conv2DLayer(net['conv4_data2'],
                                     num_filters=192,
                                     filter_size=(3,3),
                                     pad = 1)
    net['conv4'] = concat((net['conv4_part1'],net['conv4_part2']),axis=1)
    
    # conv5
    # group 2
    net['conv5_data1'] = SliceLayer(net['conv4'], indices=slice(0, 192), axis=1)
    net['conv5_data2'] = SliceLayer(net['conv4'], indices=slice(192,384), axis=1)
    net['conv5_part1'] = Conv2DLayer(net['conv5_data1'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad = 1)
    net['conv5_part2'] = Conv2DLayer(net['conv5_data2'],
                                     num_filters=128,
                                     filter_size=(3,3),
                                     pad = 1)
    net['conv5'] = concat((net['conv5_part1'],net['conv5_part2']),axis=1)
    
    # pool 5
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=(3, 3), stride = 2)

    # fc6
    net['fc6'] = DenseLayer(net['pool5'], 
                            num_units=4096,
                            nonlinearity=lasagne.nonlinearities.rectify)
    
    # fc7
    net['fc7'] = DenseLayer(
        DropoutLayer(net['fc6'], p=drop_p),
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify)
    
    # fc8 - changes: (i) num_classes, (ii) sigmoid activation
    net['fc8'] = DenseLayer(
        DropoutLayer(net['fc7'], p=drop_p),
        num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.sigmoid)
    
    return net['fc8']

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
    remainders = len(inputs) % batchsize
    if remainders!=0:
        if shuffle:
            excerpt = indices[-remainders:]
        else:
            excerpt = slice(-remainders, len(inputs))
        yield inputs[excerpt], targets[excerpt]