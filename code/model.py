#from theano.compile.nanguardmode import NanGuardMode
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

def build_model(input_var=None):
    # Input layer
    ''' 
    out: b x 3 x 227 x 227 
    '''
    lin = InputLayer(
        shape=(None, 3, 227, 227),
        input_var=input_var
        )

    # ConvPool1
    ''' 
    out: b x 96 x 27 x 27 
    out.W: 96 x 3 x 11 x 11
    '''
    """ input was b01c, need to be bc01"""
    l1c = ConvLayer(lin,
        #lasagne.layers.dimshuffle(lin, (0,3,1,2)),
        num_filters=96, filter_size=11, stride=4,
        #W = Ws['W_0'], b = bs['b_0'],
        nonlinearity=lasagne.nonlinearities.rectify
        )
    l1p = PoolLayer(
        l1c, pool_size=3, stride=2
        )

    # ConvPool2: 2 groups
    ''' 
    out: b x 256 x 13 x 13
    out.W0/1: 256 x 96 x 5 x 5
    '''
    l2c = ConvLayer(
        l1p, num_filters=256, filter_size=5, stride=1, pad=2,
        #W = Ws['W1_1'], b = bs['b1_1'],
        nonlinearity=lasagne.nonlinearities.rectify
    )
    l2p = PoolLayer(
        l2c, pool_size=3, stride=2
    )

    # Conv3
    ''' 
    out: b x 384 x 13 x 13
    out.W: 384 x 256 x 3 x 3
    '''
    l3c = ConvLayer(
        l2p, num_filters=384, filter_size=3, stride=1, pad='same',
        #W = Ws['W_2'], b = bs['b_2'],
        nonlinearity=lasagne.nonlinearities.rectify
    )

    # Conv4: 2 groups
    ''' 
    out: b x 384 x 13 x 13
    out.W0/1: 384 x 384 x 3 x 3
    '''
    l4c = ConvLayer(
        l3c, num_filters=384, filter_size=3, stride=1, pad='same',
        #W = Ws['W1_3'], b = bs['b1_3'],
        nonlinearity=lasagne.nonlinearities.rectify
    )

    # ConvPool5: 2 groups
    ''' 
    out: b x 256 x 6 x 6
    out.W0/1: 256 x 384 x 3 x 3
    '''
    l5c = ConvLayer(
        l4c, num_filters=256, filter_size=3, stride=1, pad='same',
        #W = Ws['W1_4'], b = bs['b1_4'],
        nonlinearity=lasagne.nonlinearities.rectify
    )
    l5p = PoolLayer(
        l5c, pool_size=3, stride=2
    )

    # FC6
    ''' 
    out: b x 2048 (x 1 x 1)
    out.W: 9216 x 2048
    '''
    l6 = DenseLayer(#l5p,
        DropoutLayer(l5p, p=.5),
        num_units=2048,
        #W = Ws['W_5'], b = bs['b_5'],
        nonlinearity=lasagne.nonlinearities.rectify
    )

    # FC7
    ''' 
    out: b x 2048 (x 1 x 1)
    out.W: 2048 x 2048
    '''
    l7 = DenseLayer(#l6,
        DropoutLayer(l6, p=.5),
        num_units=2048,
        #W = Ws['W_6'], b = bs['b_6'],
        nonlinearity=lasagne.nonlinearities.rectify
    )

    # FC8: replace last layer in AlexNet
    ''' 
    out: b x 22
    out.W: 2048 x 22
    '''
    l8 = DenseLayer(
        l7, num_units=22,     
        nonlinearity=lasagne.nonlinearities.softmax
    )
    return l8
   
def iterate_minibatches(inputs, targets, batchsize=96, shuffle=False): 
    assert len(inputs) == len(targets), "Input and target length mismatch"
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(num_epochs=500, mode="run", batchsize=128):  
    # Debug
    #theano.config.profile=True
    #theano.config.optimizer_profile=True
    #theano.config.warn_float64='warn'
 
    # Loading all preprocessed data
    # global Ws, bs
    Xtr, Ytr, Xva, Yva, imgMean_vals = data_prep(mode=mode)

    # Sanity check: try to overfit a tiny (eg 40 instances) subset of the data
        
    """
    COMPILING THEANO function
    """    
    start_time=time.time()
    # Prepare Theano variables for inputs and targets
    input_var = T.ftensor4('inputs')
    target_var = T.imatrix('targets')

    # Center the input images
    imgMean = T.TensorType(dtype='float32', broadcastable=(True,False,False,False))('imgMean')
    z = (input_var - imgMean)
    center_fn=theano.function([input_var, imgMean], 
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
            z
            )   

    print "\nbuilding model... "
    net0 = build_model(input_var)

    print "\ncompiling functions... "
    # Build loss function
    prediction = lasagne.layers.get_output(net0)
    loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                       target_var)
    loss = loss.mean(axis=0)

    # Create update expression for training
    # using RMSprop
    params = lasagne.layers.get_all_params(net0, 
                                           trainable=True)
    
    updates = lasagne.updates.rmsprop(loss, params, 
                                      learning_rate=1e-6, rho=0.9, epsilon=1e-08)
    '''
    updates = lasagne.updates.nesterov_momentum(loss, params, 
                                      learning_rate=0.0001, momentum=0.5)
    '''
    train_fn = theano.function([input_var, target_var], loss,
        #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
        updates=updates
        )

    ## Building loss evaluation for validation set
    va_prediction = lasagne.layers.get_output(net0, 
                                              deterministic=True)
    va_loss = lasagne.objectives.categorical_crossentropy(va_prediction,
                                                          target_var)
    va_loss = va_loss.mean(axis=0)

    va_fn = theano.function([input_var, target_var], va_loss)

    
    print("compilation finished in {:.2f}").format(
        time.time()-start_time)
    
    """
    TRAINING
    """
         
    print "\nStarting training with batchsize of %d ..." %(batchsize)    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for inputs, targets in iterate_minibatches(Xtr, Ytr, batchsize, shuffle=True):            
            inputs = center_fn(inputs, imgMean_vals)
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        va_err = 0
        va_batches = 0
        for inputs, targets in iterate_minibatches(Xva, Yva, batchsize, shuffle=True):
            inputs = center_fn(inputs, imgMean_vals)
            va_err += va_fn(inputs, targets)
            va_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(va_err / va_batches))

        '''
        # Save the model after every 5 epochs
        if epoch%5==0: 
            np.savez('Data/MSRCv2/model_'+str(epoch)+'.npz', 
                lasagne.layers.get_all_param_values(net0))
        '''
        
        '''     
        if mode=="toy" and np.allclose(train_err,0):
            print "Error for toy problem is 0. Training finished"
            break
        '''



if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['num_epochs'] = int(sys.argv[1])
    if len(sys.argv) > 2:
        kwargs['mode'] = sys.argv[2]
    
    main(**kwargs)
