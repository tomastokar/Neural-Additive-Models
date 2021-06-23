import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(preds, targets, fname = './results/roc_curve.png'):
    assert len(preds) == len(targets)

    plt.clf()
    plt.figure()
    for i in range(len(preds)):
        fpr, tpr, _ = roc_curve(targets[i], preds[i])      
        roc_auc = auc(fpr, tpr)    
        plt.plot(fpr, tpr, '-', color='orange', lw=0.5)
    
    t = np.concatenate(targets)
    p = np.concatenate(preds)
    fpr, tpr, _ = roc_curve(t, p)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, '-', color='darkorange', lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc,)
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(fname)


def plot_shape_functions(partials, features, names, nrows = 2, max_k = 10, fname = './results/shape_functions.png'):
    assert len(partials) == len(features)
    
    n = len(names)
    ncols = n // nrows
    plt.clf()
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (10, 5))
    for i in range(n):

        r = i // ncols
        c = i % ncols     

        x = np.concatenate([f[:,i] for f in features])        
        x_points = np.linspace(x.min(), x.max(), 100)        
        interpolates = []
        for j in range(len(partials)):
            x = features[j][:,i]
            y = partials[j][:,i]
            idx = x.argsort()
            x = x[idx]
            y = y[idx]
            axes[r, c].plot(x, y, '-', color = 'orange', lw = 0.5)

            interpolates.append(np.interp(x_points, x, y))

        averages = [np.average(x) for x in zip(*interpolates)]
        axes[r, c].plot(x_points, averages, '-', color = 'orange', lw = 1.5)
        
        x_, n = np.unique(x, return_counts=True) 
        twin = axes[r, c].twinx()
        if len(x_) <= max_k:
            twin.bar(x_, n, width = 1, alpha = .25)    
        else:            
            twin.hist(x, alpha = .25)
            
        axes[r, c].grid()
        axes[r, c].set_xlabel(names[i].replace('_', ' '))

        

    plt.tight_layout()
    plt.savefig(fname)




