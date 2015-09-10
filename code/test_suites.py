from data_prep import data_prep
import cv2
import numpy as np

def test_data(mode="toy"):
	X, Y, img_mean = data_prep(mode=mode)
	print("\nData statstics:")
	#idx = np.random.randint(0,100,50)
	print("{:.3f} labels per image ").format(
		Y.sum().astype("float32")/len(Y)
		)

	print("\nimg_mean statstics:\nmin: {:.2f}\nmean: {:.2f}\nmax: {:.2f}".format(
		img_mean.min(), img_mean.mean(), img_mean.max()))
	
'''
def test_iterator():
	Xtr, Ytr, Xva, Yva, img_mean,_,_ = data_prep()
	for inputs, targets in iterate_minibatches(Xva, Yva, shuffle=True):
		print(inputs.shape, targets.shape)

def test_theano():
	import theano
	import theano.tensor as T

	Xtr, Ytr, Xva, Yva, imgMean_vals,_,_ = data_prep(aug=True)

	input_var = T.ftensor4('inputs')
	target_var = T.imatrix('targets')
	imgMean = T.TensorType(dtype='float32', broadcastable=(True,False,False,False))('imgMean')
	z = input_var - imgMean
	center_fn=theano.function([input_var, imgMean], z)   

	if np.any([isinstance(x.op, T.Elemwise) for x in center_fn.maker.fgraph.toposort()]):
		print 'Used the cpu'
	else:
		print 'Used the gpu' 

	print center_fn(Xtr[:100,:,:,:], imgMean_vals)

'''
if __name__ == '__main__': 
    test_data()
    #test_iterator()
    #test_theano()