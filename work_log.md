# Automatic multi-label image annotation with Convolutional Neural Networks

## Day 0: 08/09/2016
### Foundation
- Machine Learning is about prediction/pattern recognition/... on ***data*** (images, videos, text, DNA sequence, ...), using ***models*** (which describe nature and characteristics of data) that can *learn/improve/train* themselves on the given data.
    - Refs: ML_L1-1:16 (including motivation examples); 
    - [Other example](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/): classify whether a house is in San Francisco or New York, using Decision Tree classifier. 
    This is an example of single-label single-class (SLSC) classification problem. Note: single-class problem is equivalent to binary (2-class) problem. Similarly, single-label is equivalent to binary (2-label) problem.
        

## Day 1: 13/09/2016
### Foundation
- Interpret automatic image annotation (AIA) as a multi-label multi-class (MLMC) classification problem
    - Refs:  CNN_AT/multilab_convnet-2.1
- Introduction to 2 types of approach to MLMC in AIA.
    - Refs: CNN_AT/multilab_convnet, H_CNN/girshick2014,wei2015, DATN-1.2.2.2
- High-level definition of a ***classifier*** $c(.)$: data ($x$) ${\xrightarrow{f\left(\cdot\right)}}$ feature representation ($f$) ${\xrightarrow{s\left(\cdot\right)}}$ label score ($s$) ${\xrightarrow{T\left(\cdot\right)}}$ label assignment ($\hat{y}$)
    - $\hat{y}=y_{\text{pred}}=c\left(x\right)\equiv T\circ s\circ f\left(x\right)$
    - ***Score functions*** $s(.)$ and associated ***Loss functions*** $L(.)$
        - Refs: [cs231n-LinearClassify](http://cs231n.github.io/linear-classify/)
    - ***Feature extractor $f(.)$*** can be jointly represented with score function s(.) graphically by a *standard **Neural Network*** (NN) or a *deep Neural Network* (DNN)  
        - Refs: cs231n-NN1-[quickintro](http://cs231n.github.io/neural-networks-1/#quick),[architecture](http://cs231n.github.io/neural-networks-1/#layers),[example](http://cs231n.github.io/neural-networks-1/#feedforward); DATN-2.1.2.3 <- Note: "hàm tính điểm f(x)" trong DATN tương đương với $s\circ f\left(x\right)$
- ***Learning/Training*** f(.) and s(.) is equivalent to finding *all **parameters*** $\left\{{W,b}\right\}$ that minimize L(.)
    - *Iteratively update* optimal values for {W,b} by *Mini-batch Gradient Descent* (SGD) and other learning algorithms, eg. *AdaGrad, RMSprop, Adam*
        - Refs: cs231n-[Optimization1](http://cs231n.github.io/optimization-1/#gd),[NN3-update](http://cs231n.github.io/neural-networks-3/#update)
- ***Hyper-parameters***: parameters other than {W,b}
    - Standard DNN (MLP) architecture hyperparams: activation function $\sigma(.)$, number of hidden layers, number of units in a hidden layer, 
        - Refs: cs231n-NN1-[activation](http://cs231n.github.io/neural-networks-1/#actfun),[architecture](http://cs231n.github.io/neural-networks-1/#arch)
    - Learning algorithm hyperparams:
        - update expression: learning rate, momentum, ...
            - Refs: [cs231n-NN3-update](http://cs231n.github.io/neural-networks-3/#update)
        - overall process: when to declare convergence, when to terminate training (number of epochs, early stopping criteria)
- ***Evaluating*** performance of an AIA classifier/system by standard ***metrics***, eg. precision, recall, F1, accuracy, ...
    - Ref: metrics/*
- Train on training set to find {W,b}, evaluate on validation set to choose hyperparams and avoid ***overfitting***, and evaluate on test set to get the final results on the performance.
    - Ref: slideL1-31,32 <- Note: "hàm mục tiêu f(x)" trong slideL1 tương đương với $c(x)$
    

### Implementation
1. Quick glance at Theano and Lasagne (higher-level wrapper of Theano). 
    - Alternatives: Torch (Lua), TensorFlow (Python/C++) or Caffe (C++). Try these if not comfortable with Theano after Day 2.
2. Exercises: 
    1. Digit recognization ([Lasagne example](http://lasagne.readthedocs.io/en/latest/user/tutorial.html))
        - Build a single-label multi-class (SLMC), using 3-layer NN classifer, with Lasagne. Note: we call MLP with 2 hidden layers as 3-layer NN (including scoring layer). 
        - Objective: understand code logic (how similar Lasagne API is to the foundation written on whiteboard), Theano variable declaration.
    2. Classify synthetic data to 2 classes ([Theano tutorial](http://deeplearning.net/software/theano/tutorial/examples.html##a-real-example-logistic-regression))
        - Build a single-label single-class (SLSC), using Logistic regression (LR) classifier, with raw Theano (no Lasagne here). Note: LR is equivalent to 1-layer NN i.e. NN with no hidden layer
            - Refs: [cs231n-NN1-SingleNeuron](http://cs231n.github.io/neural-networks-1/#classifier)
        - Objective: familiarize with Theano to implement user-defined functions if necessary
        

## Day 2: --/09/2016 
(est. 7-10 days after Day 1)
### Foundation
- Convolutional Neural Network (CNN): Convolutional and Pooling layers
    - Architecture hyper-parameters: number of filters, kernel size, stride, padding
- Other types of layer: DropOut, BatchNorm, Local normalization
- Tips for training: data augmentation, unit-testing, monitor training process

### Implementation
- Porting to GPU
