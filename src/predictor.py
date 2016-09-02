from scipy.optimize import minimize_scalar
import sklearn
from sklearn.metrics import f1_score
import numpy as np
import cPickle as pickle
import scipy.stats as st

class Threshold(object):
# Produce t_train for respective (labelScore, trueLabel) i.e. pairs
    def __init__(self, Xout, Y):
        self.Xout = Xout
        self.Y = Y
        self.t = []
        self.f1 = []

    def find_t_for(self):
        def minus_f1(t): 
            # we want to minimise -f1 i.e. maximise f1
            y_pred = self.Xout[i]>t
            out = f1_score(self.Y[i], y_pred)
            
            return -out
        
        for i in xrange(len(self.Y)):
            tmp = minimize_scalar(minus_f1, bounds=(0, 1), method='bounded')
            self.t.append(tmp.x)
            self.f1.append(-tmp.fun)

def produce_metrics(Y_test, y_pred, seed, num_classes, verbose=1): 
    Y_test = Y_test.astype(np.int)
    y_pred = y_pred.astype(np.int)
    # Nr of labels that appear in the test set
    K_test = np.sum(Y_test.sum(0)>0)

    # Confusion table
    tp = np.multiply(Y_test, y_pred) # true positives
    fp = y_pred - Y_test
    fp[np.where(fp<0)]=0 # false positives
    fn = Y_test - y_pred
    fn[np.where(fn<0)]=0 # false negatives
    
    # metrics
    hamming = sklearn.metrics.hamming_loss(Y_test, y_pred)
    one_error = sklearn.metrics.zero_one_loss(Y_test, y_pred) # revise definition
    coverage = sklearn.metrics.coverage_error(Y_test, y_pred)
    rank_loss = sklearn.metrics.label_ranking_loss(Y_test, y_pred)
    micro = sklearn.metrics.precision_recall_fscore_support(Y_test, y_pred, average='micro')
    macro = 0 # To depreciate
    lrap = sklearn.metrics.label_ranking_average_precision_score(Y_test, y_pred) 
    # average recall    
    B=tp.sum(1)
    C=(tp+fn).sum(1)
    ar = np.nan_to_num(B.astype(np.float)/C).mean()    
    # average f1
    if lrap!=0 and ar!=0:
        af = 2*lrap*ar/(ar+lrap)
    else: 
        af = 0       
    # per-word precision
    B=tp.sum(0)
    A=(tp+fp).sum(0)    
    C=(tp+fn).sum(0)
    wp = np.nan_to_num(B.astype(np.float)/A).sum()/K_test    
    # per-word recall
    wr = np.nan_to_num(B.astype(np.float)/C).sum()/K_test    
    # nr of recalled words
    w = np.sum(B>0)
    # AP on recalled words
    recalled_w = np.where(B>0)[0]
    Y_test_recalled = Y_test[:,recalled_w]
    y_pred_recalled = y_pred[:,recalled_w]
    ap_recalled = sklearn.metrics.label_ranking_average_precision_score(
        Y_test_recalled, y_pred_recalled) 

    metrics = [hamming, one_error, coverage, rank_loss, micro, macro, lrap, ar, af, wp, wr, w, ap_recalled]
    
    if verbose>0:
        print "Hamming Loss", hamming
        print "One-error", one_error
        print "Coverage", coverage
        print "Rank loss", rank_loss
        print "Micro metrics", micro
        # print "Macro metrics", macro
        print "Avr.Precision", lrap
        print "Avr.recall", ar
        print "Avr.F1", af
        print "Per-word.Precision", wp
        print "Per-word.recall", wr
        print "Recalled words", w
        print "AP on recalled words", ap_recalled
        """
        print "Label-specific metrics"
        print "P\tR\tF1\tName"
        for i in xrange(len(label_names)):
            print"{}\t{}\t{}\t{}".format(
                round(label[0][i],3), round(label[1][i],3), round(label[2][i],3),
            label_names[i])
        """
    return metrics
        
def predict_label(Xout, Y, t, seed, num_classes, verbose):    
    y_preds = Xout > t.reshape((len(t),1))
    return produce_metrics(Y, y_preds, seed, num_classes, verbose)

def estimate_metric(metrics, rounding=3):
    metrics = np.asarray(metrics)
    mean = metrics.mean()
    std = np.std(metrics)
    return (round(mean, rounding), round(std, rounding))

def estimate_metric_prf(metrics):
    n_runs = len(metrics)
    ps=[]; rs=[]; f1s=[]
    
    for i in xrange(n_runs):
        ps.append(metrics[i][0])
        rs.append(metrics[i][1])
        f1s.append(metrics[i][2])
    
    print estimate_metric(np.asarray(ps))
    print estimate_metric(np.asarray(rs))
    print estimate_metric(np.asarray(f1s))    

def estimate_metrics(all_metrics):
    hammings = []
    one_errors = []
    coverages = []
    rank_losses = []
    micros = []
    # macros = []
    lraps = []
    ars = []
    afs = []
    wps = []
    wrs = []
    ws = []
    aps_recalled=[]
    for run in xrange(len(all_metrics)):
        hamming = all_metrics[run][0]
        one_error = all_metrics[run][1]
        coverage = all_metrics[run][2]
        rank_loss = all_metrics[run][3]
        micro = all_metrics[run][4]
        # macro = all_metrics[run][5] # ! To depreciate
        lrap = all_metrics[run][6]
        ar = all_metrics[run][7]
        af = all_metrics[run][8]
        wp = all_metrics[run][9]
        wr = all_metrics[run][10]
        w = all_metrics[run][11]
        ap_recalled = all_metrics[run][12]
        
        hammings.append(hamming)
        one_errors.append(one_error)
        coverages.append(coverage)
        rank_losses.append(rank_loss)
        micros.append(micro)
        # macros.append(macro)
        lraps.append(lrap)
        ars.append(ar)
        afs.append(af)
        wps.append(wp)
        wrs.append(wr)
        ws.append(w)
        aps_recalled.append(ap_recalled)

    print 'Hamming loss',estimate_metric(hammings)
    print 'One-error',estimate_metric(one_errors)
    print 'Coverage',estimate_metric(coverages)
    print 'Rank loss',estimate_metric(rank_losses)
    print 'Micro\n',estimate_metric_prf(micros)
    # print 'Macro\n',estimate_metric_prf(macros)
    print 'AP',estimate_metric(lraps)
    print 'AR',estimate_metric(ars)
    print 'AF1',estimate_metric(afs)
    print 'WP',estimate_metric(wps)
    print 'WR',estimate_metric(wrs)
    print 'W',estimate_metric(ws)
    print 'Ap_recalled',estimate_metric(aps_recalled)

def wpr(Y_test, y_pred):
    '''
    Per-word precision/label    
    '''
    K_test = np.sum(Y_test.sum(0)>0)

    # Confusion table
    tp = np.multiply(Y_test, y_pred) # true positives
    fp = y_pred - Y_test
    fp[np.where(fp<0)]=0 # false positives
    fn = Y_test - y_pred
    fn[np.where(fn<0)]=0 # false negatives

    # per-word precision
    B=tp.sum(0)
    A=(tp+fp).sum(0)    
    C=(tp+fn).sum(0)
    wp = np.nan_to_num(B.astype(np.float)/A).sum()/K_test    
    # per-word recall
    wr = np.nan_to_num(B.astype(np.float)/C).sum()/K_test    

    return wp, wr

def wpr_at_k(Y_test, y_pred, k):
    '''
    Per-word precision/label at k
    '''   
    idx_k = np.where[y.sum(1)==k]
    wpr_at_k = wpr(Y_test[idx_k], y_pred[idx_k])