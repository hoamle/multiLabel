import theano
import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer
from lasagne.layers import SliceLayer, ConcatLayer
import lasagne

import os, glob, time, sys
from natsort import natsorted
import numpy as np

import cv2
import cPickle as pickle

# Load data
def load_dataset(path="Data/MSRCv2/"):
    dataset={}
    setNames = [name[-7:-4] for _,name in enumerate(glob.iglob(path+"*.npy"))]
    for name in setNames:
        dataset[name]=np.load(path+name+".npy")
        print name, dataset[name].shape, dataset[name].dtype        
    return dataset

# Data augmentation
def data_aug(mat, isinput=True, n_ops=10, isXtr=False):
    # resize, crop, mirror if "input", simply repeat if "target"
    # Compute image mean if isXtr and isInput are both True
    # image mean is computed on the original images only
    n_samples = mat.shape[0]
    
    if isinput:
        out = np.empty((n_samples*n_ops, 227, 227, 3), dtype="uint8")        
        
        # original images resized to 227x227
        for i in xrange(n_samples):            
            out[i,:,:,:] = cv2.resize(mat[i,:,:,:], (227,227))
            
        # 4 crops
        out[n_samples:2*n_samples,:,:,:] = mat[:,:227,:227,:]
        out[2*n_samples:3*n_samples,:,:,:] = mat[:,-227:,:227,:]
        out[3*n_samples:4*n_samples,:,:,:] = mat[:,:227,-227:,:]
        out[4*n_samples:5*n_samples,:,:,:] = mat[:,-227:,-227:,:]
            
        # 5 mirrors
        out[5*n_samples:,:,:,:] = out[:5*n_samples,:,::-1,:]
        
    else:
        out = np.tile(mat, (n_ops,1))
    
    if not isXtr:
        return out
    else:
        return out, np.mean(out[:n_samples,:,:,:],
            axis=0, keepdims=True, dtype="float32")

def build_model(input_var=None):
    # Input layer
    ''' 
    out: b x 227 x 227 x 3
    '''
    lin = InputLayer(
        shape=(None, 227, 227, 3),
        input_var=input_var
        )

    # ConvPool1
    ''' 
    out: b x 96 x 27 x 27 
    out.W: 96 x 3 x 11 x 11
    '''
    """ input was b01c, need to be bc01"""
    l1 = Conv2DLayer(
        lasagne.layers.dimshuffle(lin, (0,3,1,2)),
        num_filters=96, filter_size=11, stride=4,
        W = Ws['W_0'], b = bs['b_0'],
        nonlinearity=lasagne.nonlinearities.rectify
        )
    l1 = MaxPool2DLayer(
        l1, pool_size=3, stride=2
        )

    # ConvPool2: 2 groups
    ''' 
    out: b x 256 x 13 x 13
    out.W0/1: 128 x 48 x 5 x 5
    '''
    l1_0 = SliceLayer(l1, indices=slice(None,48), axis=1)
    l2_0 = Conv2DLayer(
        l1_0, num_filters=128, filter_size=5, stride=1, pad=2,
        W = Ws['W0_1'], b = bs['b0_1'],
        nonlinearity=lasagne.nonlinearities.rectify
    )
    l2_0p = MaxPool2DLayer(
        l2_0, pool_size=3, stride=2
    )

    l1_1 = SliceLayer(l1, indices=slice(48, None), axis=1)
    l2_1 = Conv2DLayer(
        l1_1, num_filters=128, filter_size=5, stride=1, pad=2,
        W = Ws['W1_1'], b = bs['b1_1'],
        nonlinearity=lasagne.nonlinearities.rectify
    )
    l2_1p = MaxPool2DLayer(
        l2_1, pool_size=3, stride=2
    )

    l2 = ConcatLayer([l2_0p,l2_1p], axis=1)

    # Conv3
    ''' 
    out: b x 384 x 13 x 13
    out.W: 384 x 256 x 3 x 3
    '''
    l3 = Conv2DLayer(
        l2, num_filters=384, filter_size=3, stride=1, pad='same',
        W = Ws['W_2'], b = bs['b_2'],
        nonlinearity=lasagne.nonlinearities.rectify
    )

    # Conv4: 2 groups
    ''' 
    out: b x 384 x 13 x 13
    out.W0/1: 192 x 192 x 3 x 3
    '''
    l3_0 = SliceLayer(l3, indices=slice(None,192), axis=1)
    l4_0 = Conv2DLayer(
        l3_0, num_filters=192, filter_size=3, stride=1, pad='same',
        W = Ws['W0_3'], b = bs['b0_3'],
        nonlinearity=lasagne.nonlinearities.rectify
    )

    l3_1 = SliceLayer(l3, indices=slice(192, None), axis=1)
    l4_1 = Conv2DLayer(
        l3_1, num_filters=192, filter_size=3, stride=1, pad='same',
        W = Ws['W1_3'], b = bs['b1_3'],
        nonlinearity=lasagne.nonlinearities.rectify
    )

    # ConvPool5: 2 groups
    ''' 
    out: b x 256 x 6 x 6
    out.W0/1: 128 x 192 x 3 x 3
    '''
    l5_0 = Conv2DLayer(
        l4_0, num_filters=128, filter_size=3, stride=1, pad='same',
        W = Ws['W0_4'], b = bs['b0_4'],
        nonlinearity=lasagne.nonlinearities.rectify
    )
    l5_0p = MaxPool2DLayer(
        l5_0, pool_size=3, stride=2
    )

    l5_1 = Conv2DLayer(
        l4_1, num_filters=128, filter_size=3, stride=1, pad='same',
        W = Ws['W1_4'], b = bs['b1_4'],
        nonlinearity=lasagne.nonlinearities.rectify
    )
    l5_1p = MaxPool2DLayer(
        l5_1, pool_size=3, stride=2
    )

    l5 = ConcatLayer([l5_0p,l5_1p], axis=1)

    # FC6
    ''' 
    out: b x 4096 (x 1 x 1)
    out.W: 9216 x 4096
    '''
    l6 = DenseLayer(
        lasagne.layers.dropout(l5, p=.5),
        num_units=4096,
        W = Ws['W_5'], b = bs['b_5'],
        nonlinearity=lasagne.nonlinearities.rectify
    )

    # FC7
    ''' 
    out: b x 4096 (x 1 x 1)
    out.W: 4096 x 4096
    '''
    l7 = DenseLayer(
        lasagne.layers.dropout(l6, p=.5),
        num_units=4096,
        W = Ws['W_6'], b = bs['b_6'],
        nonlinearity=lasagne.nonlinearities.rectify
    )

    # FC8: replace last layer in AlexNet
    ''' 
    out: b x 22
    out.W: 4096 x 22
    '''
    l8 = DenseLayer(
        l7, num_units=22,     
        nonlinearity=lasagne.nonlinearities.softmax
    )
    return l8

def data_prep(root_dir = "/home/hoa/Desktop/multiLabel/"):
    # Load data
    print("\nLoading data...")
    dataset = load_dataset()
    Xtr = dataset["Xtr"]
    Xva = dataset["Xva"]
    Ytr = dataset["Ytr"]
    Yva = dataset["Yva"]

    # Data augmentation
    '''
    img_mean=np.zeros((1,227,227,3), dtype="float32")
    '''
    print("\nAugmenting data...")    
    Xtr, img_mean = data_aug(Xtr, isXtr=True)
    Xva = data_aug(Xva)
    Ytr = data_aug(Ytr, isinput=False)
    Yva = data_aug(Yva, isinput=False)
    
    # check if load correctly
    print Xtr.shape, Xtr.dtype, Yva.shape, Yva.dtype    
    print img_mean.shape, img_mean.dtype    

    # Load pre-trained weights
    print("\nLoading pre-trained weights...")
    
    os.chdir(root_dir+'pretrained/parameters_releasing/')

    Wnames = natsorted([w[:-4] for _,w in enumerate(glob.iglob('W*'))])
    Ws = {}
    for W in Wnames:
        Ws[W] = np.load(W+".npy")
        print W, Ws[W].shape # check if load correctly
        
    bNames = natsorted([b[:-7] for _,b in enumerate(glob.iglob('b*'))])
    bs = {}
    for b in bNames:
        bs[b] = np.load(b+"_65.npy")
        print b, bs[b].shape # check if load correctly

    return Xtr, Ytr, Xva, Yva, img_mean, Ws, bs

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


def main(num_epochs=500, toy=True, batchsize=96):   
    # Loading all preprocessed data
    global img_mean, Ws, bs
    Xtr, Ytr, Xva, Yva, img_mean, Ws, bs = data_prep()

    # check if load correctly
    '''
    print Xtr.shape, Xtr.dtype, Yva.shape, Yva.dtype    
    
    for W in Ws.keys(): 
        print W, Ws[W].shape
        
    for b in bs.keys():
        print b, bs[b].shape 
    '''

    # Sanity check: try to overfit a tiny (20 instances) subset of the data
    if toy: 
        batchsize = 20
        np.random.RandomState(11)
        idx = np.random.randint(0,Xtr.shape[0]/10,batchsize)
        Xtr = Xtr[idx,:,:,:]
        Ytr = Ytr[idx,:]
        
    
    """
    COMPILING THEANO function
    """    
    start_time=time.time()
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.imatrix('targets')

    print "\nbuilding model... "
    net0 = build_model(input_var)

    print "\ncompiling functions... "
    # Build loss function
    prediction = lasagne.layers.get_output(net0)
    loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                       target_var)
    loss = loss.mean()

    # Create update expression for training
    # using RMSprop
    params = lasagne.layers.get_all_params(net0, 
                                           trainable=True)
    updates = lasagne.updates.rmsprop(loss, params, 
                                      learning_rate=1.0, rho=0.9, epsilon=1e-06)
    train_fn = theano.function([input_var, target_var], loss,
                               updates=updates)

    ## Building loss evaluation for validation set
    va_prediction = lasagne.layers.get_output(net0, 
                                              deterministic=True)
    va_loss = lasagne.objectives.categorical_crossentropy(va_prediction,
                                                          target_var)
    va_loss = va_loss.mean()

    va_fn = theano.function([input_var, target_var], va_loss)

    print("compilation finished in {:.2f}").format(
        time.time()-start_time)
    
    """
    TRAINING - HAVENT SUBTRACT IMAGE MEAN YET!!!
    """
         
    print "Starting training with batchsize of %d ..." %(batchsize)    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for inputs, targets in iterate_minibatches(Xtr, Ytr, batchsize, 
            shuffle=True):
            #inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        if not toy:
            va_err = 0
            va_batches = 0
            for inputs, targets in iterate_minibatches(Xva, Yva, batchsize, 
                shuffle=False):
                #inputs, targets = batch
                err = va_fn(inputs, targets)
                va_err += err
                va_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        if not toy:
            print("  validation loss:\t\t{:.6f}".format(va_err / va_batches))

            # Save the model after every 5 epochs
            if epoch%5==0: 
                np.savez('Data/MSRCv2/model_'+str(epoch)+'.npz', 
                    lasagne.layers.get_all_param_values(net0))
            
        if toy and np.allclose(train_err,0):
            print "Error for toy problem is 0. Training finished"
            break



if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['num_epochs'] = int(sys.argv[1])
    main(**kwargs)