import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(preds, targets, fname = './results/roc_curve.png'):
    fpr, tpr, _ = roc_curve(targets, preds)      
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.clf()
    plt.figure()
    plt.plot(fpr, tpr, '-', color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc,)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(fname)


def plot_shape_functions(p, x, names, nrows = 2, fname = './results/shape_functions.png'):
    assert p.shape == x.shape
    assert p.shape[1] == len(names)

    n = x.shape[1]    
    ncols = n // nrows
    plt.clf()
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (10, 5))
    for i in range(p.shape[1]):

        r = i // ncols
        c = i % ncols     

        axes[r, c].plot(x[:,i], p[:,i], 'o', label = 'logit')
        axes[r, c].grid()
        axes[r, c].set_xlabel(names[i].replace('_', ' '))
        axes[r, c].legend() 

        twin = axes[r, c].twinx()
        twin.hist(x[:,i], alpha = .5, label = 'hist')
        twin.legend()

    plt.tight_layout()
    plt.savefig(fname)




