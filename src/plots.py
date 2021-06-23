import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curves(results, pred_col, resp_col, fname = './results/roc_curve.png'):
    plt.clf()
    plt.figure()
    
    for _, res in results.groupby('replicate'):
        fpr, tpr, _ = roc_curve(res[resp_col], res[pred_col])      
        roc_auc = auc(fpr, tpr)    
        plt.plot(fpr, tpr, '-', color='orange', lw=0.5)

    fpr, tpr, _ = roc_curve(results[resp_col], results[pred_col])
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


def plot_shape_functions(results, features, nrows = 1, size = (8, 10), fname = './results/shape_functions.png'):
    n = len(features)
    ncols = n // nrows
    plt.clf()
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = size)
    for i, feature in enumerate(features):
        r = i // ncols
        c = i % ncols
        
        twin = axes[r, c].twinx()

        if results[feature].dtype == object:
            xlab_rot = 90
            agg_plot = 'bar'
        else:
            xlab_rot = 0
            agg_plot = 'hist'


        results.sort_values(feature, inplace = True)
        for _, res in results.groupby('replicate'):
            x = res[feature]
            y = res[feature + '_partial']
            axes[r, c].plot(x, y, '-', color = 'orange', lw = 0.25)

        x = results.pivot_table(
            index = feature, 
            columns = 'replicate', 
            values = feature + '_partial'
        )

        x = (
            x
            .interpolate()
            .mean(axis = 1)
            .sort_index()
        )

        x.plot(ax = axes[r, c], rot=xlab_rot)
        
        # Plot frequencies
        x = results[feature]
        if agg_plot == 'bar':
            x.value_counts().plot.bar(
                width = 1, 
                alpha = .15, 
                ax = twin
            )    
            twin.set_ylabel('frequnecy')

        elif agg_plot == 'hist':           
            x.plot.hist(alpha = .15, ax = twin)
                    
        axes[r, c].grid(True)
        axes[r, c].set_ylabel('Shape functions')
        axes[r, c].set_xlabel(feature.replace('_', ' '))

    plt.tight_layout()
    plt.savefig(fname)




