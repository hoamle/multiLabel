import theano
import theano.tensor as T
import lasagne

import numpy as np
import sys
import cPickle as pickle
import time
import os.path as osp
import skimage.transform

from data_prep import b01c_to_bc01, data_aug
from utils import build_model, iterate_minibatches
from predictor import Threshold, produce_metrics, estimate_metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.cross_validation import train_test_split

def main(reps, pretrained_w_path, batchsize, init_seed=0, verbose=1,
    num_classes=374, mode='ots', load_t=0, save_clf=1):
    res_root = '/home/hoa/Desktop/projects/resources'
    X_path=osp.join(res_root, 'datasets/corel5k/X_train_rgb.npy')
    Y_path=osp.join(res_root, 'datasets/corel5k/Y_train.npy')
    MEAN_IMG_PATH=osp.join(res_root, 'models/ilsvrc_2012_mean.npy')
    
        
    # baseline_msrcv2_net = build_model(pretrained_w_path, num_classes)
    
    ### LOADING DATA
    print 'LOADING DATA ...'
    X = np.load(X_path)
    Y = np.load(Y_path)
    N = len(Y)
    
    print 'Raw X,Y shape', X.shape, Y.shape
    if len(X) != len(Y):
        print 'Inconsistent number of input images and labels. X is possibly augmented.'
    
    MEAN_IMG = np.load(MEAN_IMG_PATH)
    MEAN_IMG_227 = skimage.transform.resize(
            np.swapaxes(np.swapaxes(MEAN_IMG,0,1),1,2), (227,227), mode='nearest', preserve_range=True)    
    MEAN_IMG = np.swapaxes(np.swapaxes(MEAN_IMG_227,1,2),0,1).reshape((1,3,227,227))

    
    # Prepare Theano variables for inputs
    input_var = T.tensor4('inputs')
    network = build_model(num_classes=num_classes, input_var=input_var)    
    
    layer_list = lasagne.layers.get_all_layers(network) # 22 layers
    features = lasagne.layers.get_output(layer_list[-3], # get 'fc7' in network
        deterministic=True)
    feat_fn = theano.function([input_var], features)

    def compute_feature(X, Y, batchsize=batchsize, shuffle=False):
        out = np.zeros((len(Y), 4096))
        batch_id = 0
        for batch in iterate_minibatches(X, Y, batchsize, shuffle=False):
            inputs, _ = batch
            # Flip random half of the batch
            flip_idx = np.random.choice(len(inputs),size=len(inputs)/2,replace=False)
            if len(flip_idx)>1:
                inputs[flip_idx] = inputs[flip_idx,:,:,::-1]
            # Substract mean image
            inputs = (inputs - MEAN_IMG).astype(theano.config.floatX) 
            # MEAN_IMG is broadcasted numpy-way, take note if want theano expression instead
            if len(inputs)==batchsize:
                out[batch_id*batchsize : (batch_id+1)*batchsize] = feat_fn(inputs)
                batch_id += 1
            else:
                out[batch_id*batchsize : ] = feat_fn(inputs)
                
        return out

    all_metrics = [] # store all evaluation metrics
    for seed in np.arange(reps)+init_seed:
        print '\nRUN', seed, '...'
        # Split train/val/test set
        # indicies = np.arange(len(Y))
        # Y_train_val, Y_test, idx_train_val, idx_test = train_test_split(
        #     Y, indicies, random_state=seed, train_size=float(2)/3)
        # # Y_train, Y_val, idx_train, idx_val = train_test_split(
        #     Y_train_val, idx_train_val, random_state=seed)
        
        # print "Train/val/test set size:",len(idx_train),len(idx_val),len(idx_test)

        # idx_aug_train = data_aug(idx_train, mode='aug', isMat='idx')
        # Xaug_train = X[idx_aug_train]
        # Yaug_train = data_aug(Y_train, mode='aug', isMat='Y')

        # idx_aug_val = data_aug(idx_val, mode='aug', isMat='idx')
        # Xaug_val = X[idx_aug_val]
        # Yaug_val = data_aug(Y_val, mode='aug', isMat='Y')

        # Module 2 training set is composed of module 1 training and validation set 
        idx_train_val = np.arange(len(Y))
        # idx_aug_train_val = data_aug(idx_train_val, mode='aug', isMat='idx')
        # Xaug_train_val = X[idx_aug_train_val]
        # Yaug_train_val = data_aug(Y, mode='aug', isMat='Y')
        Xaug_train_val = data_aug(X, mode='noaug', isMat='X', N=N)
        if Xaug_train_val.shape[1]!=3:
            Xaug_train_val = b01c_to_bc01(Xaug_train_val)

        Yaug_train_val = Y

        # Test set
        X_test = np.load(osp.join(res_root,'datasets/corel5k/X_test_rgb.npy'))
        if X_test.shape[1]!=3:
            X_test = b01c_to_bc01(X_test)
        Y_test = np.load(osp.join(res_root,'datasets/corel5k/Y_test.npy'))

        # load reference_net
        f = open(pretrained_w_path, 'r')
        w_list = pickle.load(f) # list of 11 (W,b)-pairs
        f.close()
        
        # Reset init weights
        lasagne.layers.set_all_param_values(layer_list[-3], w_list[:-2]) 
        # exclude (W,b) of fc8
        # BIG NOTE: don't be confused, it's pure coincident that layer_list 
        # and w_list have the same index here. The last element of layer_list are 
        # [.., fc6, drop6, fc7, drop7, fc8], while w_list are 
        # [..., W, b, W, b, W, b] which, eg w_list[-4] and w_list[-3] correspond to
        # params that are associated with fc7 i.e. params that connect drop6 to fc7
                    
        ### Extracting features on fc7
        feats_train = compute_feature(Xaug_train_val, Yaug_train_val)

        if mode=="ots":            
            # OvR linear SVM classifier
            start_time = time.time()            
            clf_path = '../snapshot_models/{0}{1}{2}.pkl'.format(num_classes,mode,seed)
            if osp.exists(clf_path):
                save_clf = 0                
                with open(clf_path, 'rb') as fid:
                    clf = pickle.load(fid)
                print 'Loaded', clf_path 
            else:
                clf = OneVsRestClassifier(LinearSVC())
                clf.fit(feats_train, Yaug_train_val)

            if save_clf:
                with open(clf_path, 'wb') as fid: 
                # save classifier
                    pickle.dump(clf, fid) 
            
            # Prediction on test set    
            start_time = time.time()
            
            # Feature extraction on test set
            feats_test = compute_feature(X_test, Y_test)
            y_pred = clf.predict(feats_test)
            print('Prediction on test set: {:.1f}s').format(time.time()-start_time)    

        elif mode=="tune": # Module 2 of CNN-AT, only train the label scorer
            print "MODULE 2"
            clf = OneVsRestClassifier(LogisticRegression(C=2000)) # C=1/5e-4
            clf.fit(feats_train, Yaug_train_val)
            score_train = clf.predict_proba(feats_train)

            # LABEL THRESHOLDER
            if not load_t:
                start_time = time.time()                        
                thresholds = Threshold(score_train, Yaug_train_val)
                thresholds.find_t_for() # determine t_train for each score_train. It will take a while
                t_train = np.asarray(thresholds.t)
                print 't_train is in ', t_train.min(), '..', t_train.max() 
                # `thresholds` holds t_train vector in .t attribute
                print('t_train produced in {:.3f}s').format(time.time()-start_time)
                np.save(osp.join('t', "{0}tune{1}.npy".format(num_classes,seed)), t_train)
            else:
                print 'Loading t_train in {0}tune{1}.npy'.format(num_classes,seed)
                t_train = np.load(osp.join('t', "{0}tune{1}.npy".format(num_classes,seed)))

            # ## Ridge regression for predicting t
            regr = RidgeCV(cv=5) 
            # Ridge() is LinearClassifier() with L2-reg
            regr.fit(score_train, t_train) 


            # TESTING PHASE
            start_time = time.time()
            feats_test = compute_feature(X_test, Y_test)
            score_test = clf.predict_proba(feats_test)
            t_test = regr.predict(score_test)
            print 'original t_test is in ', min(t_test), '..', max(t_test)
            epsilon = 1e-6
            t_test[t_test>1] = max(t_test[t_test<1]) - epsilon
            t_test[t_test<0] = 0 # ! Keep t_test in [0,1]
            print 'corrected t_test is in ', min(t_test), '..', max(t_test) 

            y_pred = score_test > t_test.reshape((len(t_test),1))

        # Evaluate
        k=5
        if k: # Evaluate@k
            idx_k = np.where(y_pred.sum(1)==k) # Extract examples annotated by exactly k labels
            Y_test = Y_test[idx_k]
            y_pred = y_pred[idx_k]
            print "Nr. of test images: %d" %len(idx_k[0])

        metrics = produce_metrics(Y_test, y_pred, seed, num_classes, verbose=verbose)
        all_metrics.append(metrics)
        
        



    print '\nFINAL ESTIMATES FOR {0} IN {1} RUNS'.format(mode, len(all_metrics))
    estimate_metrics(all_metrics)
    np.save(osp.join('metrics',"{0}{1}_allmetrics.npy".format(num_classes,mode)), all_metrics)
            
if __name__ == '__main__':
    kwargs = {}    
    if len(sys.argv) > 1:
        kwargs['reps'] = int(sys.argv[1])
    if len(sys.argv) > 2:
        kwargs['batchsize'] = int(sys.argv[2])
    if len(sys.argv) > 3:
        kwargs['pretrained_w_path'] = sys.argv[3] 
    if len(sys.argv) > 4:
        kwargs['mode'] = sys.argv[4] 
    if len(sys.argv) > 5:
        kwargs['load_t'] = int(sys.argv[5])
    # if len(sys.argv) > 5:
    #     kwargs['init_seed'] = int(sys.argv[5])    

    main(**kwargs)