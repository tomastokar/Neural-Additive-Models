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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(fname)