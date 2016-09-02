# coding: utf-8
#import os    
#os.environ['THEANO_FLAGS'] = "device=gpu1" # for running inside ipython notebook

import theano
import theano.tensor as T
import lasagne
from lasagne.utils import floatX
import numpy as np

from utils import build_model, iterate_minibatches
from data_prep import b01c_to_bc01, data_aug
from predictor import Threshold, predict_label, estimate_metrics

import time, sys
from sklearn.cross_validation import train_test_split
from visualize import plot_loss, plot_conv_weights
import skimage.transform
from sklearn import linear_model
import os.path as osp
import cPickle as pickle

def main(reps, pretrained_w_path, do_module1, init_seed=0, load_t=0, num_epochs=200,
    batchsize=96, fine_tune=0, patience=500, lr_init = 1e-3, optim='adagrad', toy=0,
    num_classes=374):
    res_root = '/home/hoa/Desktop/projects/resources'
    X_path=osp.join(res_root, 'datasets/corel5k/X_train_rgb.npy')
    Y_path=osp.join(res_root, 'datasets/corel5k/Y_train.npy')
    MEAN_IMG_PATH=osp.join(res_root, 'models/ilsvrc_2012_mean.npy')
    snapshot=50 # save model after every `snapshot` epochs
    
    drop_p=0.5 # drop out prob.
    lambda2=0.0005/2 # l2-regularizer constant    
    # step=patience/4 # decay learning after every `step` epochs
    lr_patience=60 # for learning rate schedule, if optim=='momentum'    
    if toy: # unit testing
        num_epochs=10
        data_multi=3
        reps = 2        
        #drop_p=0
        #lambda2=0
    
    # Create name tag for the experiment
    if fine_tune:
        full_or_tune = 'tune' # description tag for storing associated files
    else:
        full_or_tune = 'full'
    time_stamp=time.strftime("%y%m%d%H%M%S", time.localtime()) 
    snapshot_root = '../snapshot_models/'
    
    # LOADING DATA
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

    all_metrics = [] # store metrics in each run
    time_profiles = {
    'train_module1': [],
    'train_module1_eff': [],
    'train_module2': [],
    'test': []
    } # record training and testing time
   
     # PREPARE THEANO EXPRESSION FOR BOTH MODULES
    print 'COMPILING THEANO EXPRESSION ...'
    input_var = T.tensor4('inputs')
    target_var = T.imatrix('targets')        
    network = build_model(num_classes=num_classes, input_var=input_var)    

    # Create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var) 
    weights = lasagne.layers.get_all_params(network, regularizable=True)
    l2reg = theano.shared(floatX(lambda2))*T.sum([T.sum(w ** 2) for w in weights])
    loss = loss.mean() + l2reg
    
    lr = theano.shared(np.array(lr_init, dtype=theano.config.floatX))
    lr_decay = np.array(1./3, dtype=theano.config.floatX)
    
    # Create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    # last-layer case is actually very simple:
    # `params` above is a list of all (W,b)-pairs
    # Therefore last layer's (W,b) is params[-2:]
    if fine_tune == 7: # tuning params from fc7 to fc8
        params = params[-2:] 
    # elif fine_tune == 6: # tuning params from fc6 to fc8
    #     params = params[-4:]
    # TODO adjust for per-layer training with local_lr
    
    if optim=='momentum':
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9) 
    elif optim=='rmsprop':
        updates = lasagne.updates.rmsprop(loss, params, learning_rate=lr, rho=0.9, epsilon=1e-06) 
    elif optim=='adam':
        updates = lasagne.updates.adam(
            loss, params, learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
    elif optim=='adagrad':
        updates = lasagne.updates.adagrad(loss, params, learning_rate=lr, epsilon=1e-06)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean() + l2reg
    # zero-one loss with threshold t = 0.5 for reference
    # zero_one_loss = T.abs_((test_prediction > theano.shared(floatX(0.5))) - target_var).sum(axis=1)
    #zero_one_loss /= target_var.shape[1].astype(theano.config.floatX)
    #zero_one_loss = zero_one_loss.mean()
    
    # Compile a function performing a backward pass (training step)  on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    bwd_fn = theano.function([input_var, target_var], loss, updates=updates,)
    # Compile a second function performing a forward pass, 
    # returns validation loss, 0/1 Error, score i.e. Xout:
    fwd_fn = theano.function([input_var, target_var], test_loss)

    # Create a theano function for computing score
    score = lasagne.layers.get_output(network, deterministic=True)
    score_fn = theano.function([input_var], score)

    def compute_score(X, Y, batchsize=batchsize, shuffle=False):
        out = np.zeros(Y.shape)
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
                out[batch_id*batchsize : (batch_id+1)*batchsize] = score_fn(inputs)
                batch_id += 1
            else:
                out[batch_id*batchsize : ] = score_fn(inputs)
                
        return out

    try:
        #  MAIN LOOP FOR EACH RUN    
        for seed in np.arange(reps)+init_seed:
            snapshot_name = str(num_classes)+'alex'+time_stamp+full_or_tune+str(seed)
            # reset learning rate
            lr.set_value(lr_init)

            print '\nRUN', seed, '...'
            # Split train/val/test set
            # indicies = np.arange(len(Y))
            # Y_train_val, Y_test, idx_train_val, idx_test = train_test_split(
            #     Y, indicies, random_state=seed, train_size=float(2)/3)
            idx_train_val = np.arange(len(Y))
            
            # Module 2 training set is composed of module 1 training and validation set 
            idx_aug_train_val = data_aug(idx_train_val, mode='aug', isMat='idx')
            Xaug_train_val = data_aug(X, mode='noaug', isMat='X')
            if Xaug_train_val.shape[1]!=3:
                Xaug_train_val = b01c_to_bc01(Xaug_train_val)
            Yaug_train_val = Y

            # train/val/test set for module 1
            Y_train, Y_val, idx_train, idx_val = train_test_split(
                Y, idx_train_val, random_state=seed)
            
            idx_aug_train = idx_train
            Xaug_train = Xaug_train_val[idx_aug_train]
            Yaug_train = Y_train

            idx_aug_val = idx_val
            Xaug_val = Xaug_train_val[idx_aug_val]
            Yaug_val = Y_val

            # Test set
            X_test = np.load(osp.join(res_root,'datasets/corel5k/X_test_rgb.npy'))
            if X_test.shape[1]!=3:
                X_test = b01c_to_bc01(X_test)
            Y_test = np.load(osp.join(res_root,'datasets/corel5k/Y_test.npy'))

            print "Augmented train/val/test set size:",len(Xaug_train),len(Yaug_val), len(X_test)
            print "Augmented (X,Y) dtype:", Xaug_train.dtype, Yaug_val.dtype
            print "Processed Mean image:",MEAN_IMG.dtype,MEAN_IMG.shape

            if toy: # try to overfit a tiny subset of the data
                Xaug_train = Xaug_train[:batchsize*data_multi + batchsize/2]
                Yaug_train = Yaug_train[:batchsize*data_multi + batchsize/2]
                Xaug_val = Xaug_val[:batchsize + batchsize/2]
                Yaug_val = Yaug_val[:batchsize + batchsize/2]

            # Init by pre-trained weights, if any
            if len(pretrained_w_path)>0:
                layer_list = lasagne.layers.get_all_layers(network) # 22 layers
                if pretrained_w_path.endswith('pkl'): 
                # load reference_net
                # use case: weights initialized from pre-trained reference nets                
                    f = open(pretrained_w_path, 'r')
                    w_list = pickle.load(f) # list of 11 (W,b)-pairs
                    f.close()
                    
                    lasagne.layers.set_all_param_values(layer_list[-3], w_list[:-2]) 
                    # exclude (W,b) of fc8
                    # BIG NOTE: don't be confused, it's pure coincident that layer_list 
                    # and w_list have the same index here. The last element of layer_list are 
                    # [.., fc6, drop6, fc7, drop7, fc8], while w_list are 
                    # [..., W, b, W, b, W, b] which, eg w_list[-4] and w_list[-3] correspond to
                    # params that are associated with fc7 i.e. params that connect drop6 to fc7
                    
                    
                elif pretrained_w_path.endswith('npz'): 
                # load self-trained net 
                # use case: continue training from a snapshot model
                    with np.load(pretrained_w_path) as f: # NOTE: only load snapshot of the same `seed`
                        w_list = [f['arr_%d' % i] for i in range(len(f.files))] 
                    lasagne.layers.set_all_param_values(network, w_list)

                elif pretrained_w_path.endswith('/'): # init from 1 of the 30 snapshots
                    from os import listdir
                    import re
                    files = [f for f in listdir(pretrained_w_path) if osp.isfile(osp.join(pretrained_w_path, f))]
                    for file_name in files:
                        regex_seed = 'full%d_' %seed
                        match_seed = re.search(regex_seed, file_name)
                        if match_seed:
                            regex = r"\d+[a-zA-Z]+\d+[a-zA-Z]+\d+\_\d+"
                            match = re.search(regex, file_name)
                            snapshot_name = match.group(0)
                            print snapshot_name
                            with np.load(osp.join(pretrained_w_path,snapshot_name)+'.npz') as f: 
                                w_list = [f['arr_%d' % i] for i in range(len(f.files))] 
                            lasagne.layers.set_all_param_values(network, w_list)

            # START MODULE 1
            module1_time = 0
            if do_module1:
                print 'MODULE 1' 
                training_history={}
                training_history['iter_training_loss'] = []
                training_history['iter_validation_loss'] = []
                training_history['training_loss'] = []
                training_history['validation_loss'] = []
                training_history['learning_rate'] = []
                
                # http://deeplearning.net/tutorial/gettingstarted.html#early-stopping
                # early-stopping parameters
                n_train_batches = Xaug_train.shape[0] / batchsize
                if Xaug_train.shape[0] % batchsize != 0:
                    n_train_batches += 1
                patience = patience  # look as this many examples regardless
                patience_increase = 2     # wait this much longer when a new best is found
                lr_patience_increase = 1.01
                improvement_threshold = 0.995  # a relative improvement of this much is
                                               # considered significant; a significant test
                                               # MIGHT be better
                validation_frequency = min(n_train_batches, patience/2)
                                              # go through this many
                                              # minibatches before checking the network
                                              # on the validation set; in this case we
                                              # check every epoch
                best_params = None
                epoch_validation_loss = 0 # indicates that valid_loss has not been computed yet
                best_validation_loss = np.inf
                best_iter = -1
                lr_iter = -1
                test_score = 0.
                start_time = time.time()
                done_looping = False
                epoch = 0
                
                # Finally, launch the training loop.
                print("Starting training...")
                # We iterate over epochs:
                print("\nEpoch\tTrain Loss\tValid Loss\tBest-ValLoss-and-Iter\tTime\tL.Rate")
                sys.setrecursionlimit(10000)

                try: # Early-stopping implementation
                    while (not done_looping) and (epoch<num_epochs):
                        # In each epoch, we do a full pass over the training data:
                        train_err = 0
                        train_batches = 0
                        start_time = time.time()
                        for batch in iterate_minibatches(Xaug_train, Yaug_train, batchsize, shuffle=True):
                            inputs, targets = batch
                            # Horizontal flip half of the images
                            bs = inputs.shape[0]
                            indices = np.random.choice(bs, bs / 2, replace=False)
                            inputs[indices] = inputs[indices, :, :, ::-1]
                            
                            # Substract mean image
                            inputs = (inputs - MEAN_IMG).astype(theano.config.floatX) 
                            # MEAN_IMG is broadcasted numpy-way, take note if want theano expression instead
                    
                            train_err_batch = bwd_fn(inputs, targets) 
                            train_err += train_err_batch            
                            train_batches += 1
                            
                            iter_now = epoch*n_train_batches + train_batches
                            training_history['iter_training_loss'].append(train_err_batch)
                            training_history['iter_validation_loss'].append(epoch_validation_loss)
                            
                            if (iter_now+1) % validation_frequency == 0:
                                # a full pass over the validation data:       
                                val_err = 0
                                #zero_one_err = 0
                                val_batches = 0
                                for batch in iterate_minibatches(Xaug_val, Yaug_val, batchsize, shuffle=False):
                                    inputs, targets = batch
                                    # Substract mean image
                                    inputs = (inputs - MEAN_IMG).astype(theano.config.floatX) 
                                    # MEAN_IMG is broadcasted numpy-way, take note if want theano expression instead
                                    
                                    val_err_batch = fwd_fn(inputs, targets)
                                    val_err += val_err_batch
                                    val_batches += 1                
                                epoch_validation_loss = val_err / val_batches
                                if epoch_validation_loss < best_validation_loss:
                                    if epoch_validation_loss < best_validation_loss*improvement_threshold:
                                        patience = max(patience, iter_now * patience_increase)
                                        # lr_patience *= lr_patience_increase
                                        
                                    best_params = lasagne.layers.get_all_param_values(network)
                                    best_validation_loss = epoch_validation_loss
                                    best_iter = iter_now
                                    lr_iter = best_iter


                                else: # decay learning rate if optim=='momentum'
                                    if optim=='momentum' and (iter_now - lr_iter) >  lr_patience:
                                        lr.set_value(lr.get_value() * lr_decay) 
                                        lr_iter = iter_now
                            
                            if patience <= iter_now:
                                done_looping = True
                                break
                        
                        # Record training history
                        training_history['training_loss'].append(train_err / train_batches)
                        training_history['validation_loss'].append(epoch_validation_loss)
                        training_history['learning_rate'].append(lr.get_value())

                        epoch_time = time.time() - start_time
                        module1_time += epoch_time
                        # Then we print the results for this epoch:
                        print("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\t{:.3f}\t{}".format(
                                epoch+1, 
                                training_history['training_loss'][-1],
                                training_history['validation_loss'][-1],
                                best_validation_loss,
                                best_iter+1,
                                epoch_time,
                                training_history['learning_rate'][-1]
                            ))
                        
                        if (epoch+1)%snapshot==0: # TODO try to save weights at best_iter
                            snapshot_path_string = snapshot_root+snapshot_name+'_'+str(iter_now+1)
                            try: # use case: terminate experiment before reaching `reps`
                                np.savez(snapshot_path_string+'.npz', *best_params)
                                np.savez(snapshot_path_string+'_history.npz', training_history)
                                plot_loss(training_history, snapshot_path_string+'_loss.png')
                                # plot_conv_weights(lasagne.layers.get_all_layers(network)[1], 
                                #     snapshot_path_string+'_conv1weights_')
                            except KeyboardInterrupt, TypeError:
                                print 'Did not save', snapshot_name+'_'+str(iter_now+1)
                                pass

                        epoch += 1

                except KeyboardInterrupt, MemoryError: # Sadly this can only catch KeyboardInterrupt
                    pass
                print 'Training finished or KeyboardInterrupt (Training is never finished, only abandoned)'
                
                module1_time_eff = module1_time / iter_now * best_iter 
                print('Total and Effective training time are {:.0f} and {:.0f}').format(
                    module1_time, module1_time_eff)
                time_profiles['train_module1'].append(module1_time)
                time_profiles['train_module1_eff'].append(module1_time_eff)
                
                # Save model after num_epochs or KeyboardInterrupt
                if (epoch+1)%snapshot!=0: # to avoid duplicate save
                    snapshot_path_string = snapshot_root+snapshot_name+'_'+str(iter_now+1)
                    if not toy:
                        try: # use case: terminate experiment before reaching `reps`
                            print 'Saving model...'
                            np.savez(snapshot_path_string+'.npz', *best_params)
                            np.savez(snapshot_path_string+'_history.npz', training_history)
                            plot_loss(training_history, snapshot_path_string+'_loss.png')
                            # plot_conv_weights(lasagne.layers.get_all_layers(network)[1], 
                            #     snapshot_path_string+'_conv1weights_')
                        except KeyboardInterrupt, TypeError:
                            print 'Did not save', snapshot_name+'_'+str(iter_now+1)
                            pass
                # And load them again later on like this:
                #with np.load('../snapshot_models/23alex16042023213910.npz') as f:
                #    param_values = [f['arr_%d' % i] for i in range(len(f.files))] # or
                #    training_history = f['arr_0'].items()
                # lasagne.layers.set_all_param_values(network, param_values)                
            
            # END OF MODULE 1             
                
            # START MODULE 2
            print '\nMODULE 2' 
            if not do_module1:
                if pretrained_w_path.endswith('pkl'):
                    snapshot_name = str(num_classes)+'alexOTS' # short for "off-the-shelf init"
                
                elif pretrained_w_path.endswith('npz'): # Resume from a SINGLE snapshot
                    # extract name pattern, e.g. '23alex16042023213910full10' 
                    # from string '../snapshot_models/23alex16042023213910full10_100.npz'
                    import re
                    regex = r"\d+[a-zA-Z]+\d+[a-zA-Z]+\d+"
                    match = re.search(regex, pretrained_w_path)
                    snapshot_name = match.group(0)
                
                elif pretrained_w_path.endswith('/'): # RESUMED FROM TRAINED MODULE 1 (ONE-TIME USE)
                    from os import listdir
                    import re
                    files = [f for f in listdir(pretrained_w_path) if osp.isfile(osp.join(pretrained_w_path, f))]
                    for file_name in files:
                        regex_seed = 'full%d_' %seed
                        match_seed = re.search(regex_seed, file_name)
                        if match_seed:
                            regex = r"\d+[a-zA-Z]+\d+[a-zA-Z]+\d+\_\d+"
                            match = re.search(regex, file_name)
                            snapshot_name = match.group(0)
                            print snapshot_name
                            with np.load(osp.join(pretrained_w_path,snapshot_name)+'.npz') as f: 
                                w_list = [f['arr_%d' % i] for i in range(len(f.files))] 
                            lasagne.layers.set_all_param_values(network, w_list)

            else: # MAIN BRANCH - assume do_module1 is True AND have run `snapshot` epochs
                if (epoch+1)>snapshot: 
                    with np.load(snapshot_path_string+'.npz') as f: # reload the best params for module 1 
                        w_list = [f['arr_%d' % i] for i in range(len(f.files))] 
                    lasagne.layers.set_all_param_values(network, w_list)
           
            score_train = compute_score(Xaug_train_val, Yaug_train_val)
            start_time = time.time()

            if load_t:                 
                from os import listdir
                import re
                if not pretrained_w_path.endswith('/'):
                    files = [pretrained_w_path]
                else:
                    files = [f for f in listdir(pretrained_w_path) if osp.isfile(osp.join(pretrained_w_path, f))]
                for file_name in files:
                    regex_seed = '{0}{1}'.format(full_or_tune,seed)
                    match_seed = re.search(regex_seed, file_name)
                    if match_seed:
                        regex = r"\d+[a-zA-Z]+\d+[a-zA-Z]+\d+\_\d+"
                        match = re.search(regex, file_name)
                        snapshot_name = match.group(0)
                        t_train = np.load(osp.join('t','{0}.npy'.format(snapshot_name)))

            else: # MAIN BRANCH
                thresholds = Threshold(score_train, Yaug_train_val)
                thresholds.find_t_for() # determine t_train for each score_train. It will take a while
                t_train = np.asarray(thresholds.t)
                print 't_train is in ', t_train.min(), '..', t_train.max() 
                # `thresholds` holds t_train vector in .t attribute
                print('t_train produced in {:.3f}s').format(time.time()-start_time)
                np.save('t/'+snapshot_name+'.npy', t_train)

            
            # Predictive model for t
            regr = linear_model.RidgeCV(cv=5) 
            # Ridge() is LinearClassifier() with L2-reg
            regr.fit(score_train, t_train) 

            time_profiles['train_module2'].append(time.time()-start_time)
            # END OF MODULE 2        

            # TESTING PHASE
            start_time = time.time()
            score_test = compute_score(X_test, Y_test)
            t_test = regr.predict(score_test)
            print 'original t_test is in ', min(t_test), '..', max(t_test)
            t_test[t_test>1] = max(t_test[t_test<1])
            t_test[t_test<0] = min(t_test[t_test>0]) # ! Keep t_test in [0,1]
            print 'corrected t_test is in ', min(t_test), '..', max(t_test) 
        
        # Predict label 
        metrics = predict_label(score_test, Y_test, t_test, seed, num_classes, verbose=1)        
        time_profiles['test'].append(time.time()-start_time)

        all_metrics.append(metrics)
        

    except KeyboardInterrupt:
        pass

    # Store all_metrics and print estimates
    np.save(osp.join('metrics',snapshot_name + '_allmetrics.npy'), all_metrics)
    np.save(osp.join('metrics',snapshot_name + '_time'), time_profiles)

    print '\nFINAL ESTIMATES FOR', snapshot_name,'IN',len(all_metrics),'RUNS'
    estimate_metrics(all_metrics)

    
if __name__ == '__main__':
    kwargs = {}    
    if len(sys.argv) > 1:
        kwargs['reps'] = int(sys.argv[1]) # number of repeats 
    if len(sys.argv) > 2:
        kwargs['batchsize'] = int(sys.argv[2]) 
    if len(sys.argv) > 3:
        kwargs['fine_tune'] = int(sys.argv[3]) # 1 (Yes) or 0 (only tune last layer)
    if len(sys.argv) > 4:
        kwargs['pretrained_w_path'] = sys.argv[4] # initialize with pretrained params (.pkl/.npz file)
                                                  # or "none" (random weights)
    if len(sys.argv) > 5:
        kwargs['do_module1'] = int(sys.argv[5]) # 1 (Yes) or 0 (run module 2 only)
    if len(sys.argv) > 6:
        kwargs['load_t'] = int(sys.argv[6]) # use pre-generated t_train?
    if len(sys.argv) > 7:
        kwargs['toy'] = int(sys.argv[7]) 
        # set to 1 to unit test on a tiny subset of the data
    if len(sys.argv) > 8:
        kwargs['init_seed'] = int(sys.argv[8]) # resume from which `seed`?        
    main(**kwargs)
