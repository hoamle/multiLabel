import numpy as np
import skimage.transform

# Data augmentation
def b01c_to_bc01(X):
    return np.swapaxes(np.swapaxes(X,2,3), 1, 2)    

def data_aug(mat, mode, isMat, N):
    n_samples = len(mat)
    
    if mode=="aug":
        n_ops = 5
    else: # mode=='noaug'
        n_ops = 1    
            
    if isMat=='X':
        out = np.empty((n_samples*n_ops, 227, 227, 3), dtype="uint8")     
        
        for i in xrange(n_samples):             
            out[i] = skimage.transform.resize(
                mat[i], (227,227), mode='nearest', preserve_range=True)           
                
        if mode=="aug":            
            # 4 corner-crops
            out[n_samples:2*n_samples,:,:,:] = mat[:,:227,:227,:]
            out[2*n_samples:3*n_samples,:,:,:] = mat[:,-227:,:227,:]
            out[3*n_samples:4*n_samples,:,:,:] = mat[:,:227,-227:,:]
            out[4*n_samples:5*n_samples,:,:,:] = mat[:,-227:,-227:,:]
                
            # 5 mirrors
            #out[5*n_samples:,:,:,:] = out[:5*n_samples,:,::-1,:]        
            
    #     if b01c:
    #         out = b01c_to_bc01(out)
    if isMat=='idx':
        out = np.tile(mat, n_ops)
                                              
        if mode=="aug":
            for k in xrange(1,n_ops):
                out[k*n_samples:(k+1)*n_samples] = mat+N*k
                
    elif isMat=='Y':
        out = np.tile(mat, (n_ops,1))           
        
    return out