from sklearn.cross_validation import train_test_split
import os, glob
from natsort import natsorted
import numpy as np
import cv2

# Data augmentation
def b01c_to_bc01(X):
    b,row,col,c = X.shape
    out = np.zeros((b,c,row,col), dtype="uint8")
    for i in xrange(c):
        out[:,i:,:] = np.expand_dims(X[:,:,:,i], 1)

    return out

def data_aug(mat, isinput=True, n_ops=10, b01c=True, mode="run"):
    # resize, crop, mirror if "input", simply tile if "target"
    # Compute image mean if isXtr and isInput are both True
    # image mean is computed on the original images only
    n_samples = len(mat)
    
    if isinput:
        if mode!="toy":            
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
            start_idx=14
            out = mat[:,start_idx:start_idx+227,start_idx:start_idx+227,:]
            
        if b01c:
            out = b01c_to_bc01(out)
        
    else:
        if mode=="toy":
            n_ops=1
        out = np.tile(mat, (n_ops,1))
    
    return out
'''
# Split training and validation set
def 

i = np.arange(X.shape[0])
Xtr, Xva, Ytr, Yva = train_test_split(X, Y, train_size = 0.75, random_state=11)
print " %d training examples and %d validation images" % (Xtr.shape[0], Xva.shape[0])
'''
def data_prep(root_dir = "/home/hoa/Desktop/multiLabel/", mode="run"):
    # Load data
    print("\nLoading data...")
    X = np.load("Data/MSRCv2/X.npy")
    Y = np.load("Data/MSRCv2/Y.npy")    
    
    # Data augmentation
    print("\nAugmenting data...")    
    X = data_aug(X, mode=mode)
    Y = data_aug(Y, mode=mode, isinput=False).astype("float32")
    print X.shape, X.dtype, Y.shape, Y.dtype    # check if load correctly

    # Compute Mean image
    if mode=="toy": 
        n_ops=1
    else: 
        n_ops=10
    img_mean=np.mean(X[:len(X)/n_ops,:,:,:],
        axis=0, keepdims=True, dtype="float32")        
    
    print img_mean.shape, img_mean.dtype    # check if load correctly
    
    ''' DEPRECIATED
    # Load pre-trained weights
    print("\nLoading pre-trained weights...")        
    os.chdir(root_dir+'pretrained/parameters_releasing/')

    Wnames = natsorted([w[:-4] for _,w in enumerate(glob.iglob('W*'))])
    Ws = {}
    for W in Wnames:
        Ws[W] = np.load(W+".npy")
        #print W, Ws[W].shape # check if load correctly
        
    bNames = natsorted([b[:-7] for _,b in enumerate(glob.iglob('b*'))])
    bs = {}
    for b in bNames:
        bs[b] = np.load(b+"_65.npy")
        #print b, bs[b].shape # check if load correctly
    os.chdir(root_dir)
    '''
    return X, Y, img_mean