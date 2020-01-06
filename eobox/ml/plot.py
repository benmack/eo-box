import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_confusion_matrix(cm, 
                          class_names=None,
                          switch_axes=False,
                          vmin=None, 
                          vmax=None, 
                          cmap='RdBu',
                          robust=True, 
                          logscale_color=False, 
                          fmt=",", 
                          annot_kws=None,
                          cbar=False,
                          mask_zeros=False,
                          ax=None):
    """Plot a confusion matrix with the precision and recall added.
    
    TODO: documentation ...

    TODO: check https://github.com/wcipriano/pretty-print-confusion-matrix

    switch_axes if False the CM is returned as is the default of sklearn 
    with the rows being the actual class and the columns the predicted class
    if True, the axis are switched.

    """
    import matplotlib
    
    normalize=False
    
    ax = ax or plt.gca()
    
    if class_names is None:
        class_names = range(1, cm.shape[0] + 1)
        
    
    diagonal = cm.diagonal()
    n_samples = cm.sum().sum()
    col_wise_sums = cm.sum(axis=0)
    row_wise_sums = cm.sum(axis=1)

    precision = np.round((diagonal / row_wise_sums * 100), 2).astype(cm.dtype)
    recall = np.round((diagonal /  col_wise_sums * 100), 0).astype(cm.dtype)
    precision_dim_expanded = np.expand_dims(np.concatenate((precision, [0])), axis=1)
    overall_accuracy = np.round((diagonal.sum() / n_samples) * 100, 0).astype(cm.dtype)
    cm = np.concatenate([cm, np.expand_dims(recall, axis=0)], axis=0)
    cm = np.concatenate([cm, precision_dim_expanded], axis=1)
    cm_annot = cm.copy()
    cm_annot[-1, -1] = overall_accuracy
    
    if logscale_color:
        cm = np.log(cm)
        cm[cm == -np.inf] = 0
    
    # make only off-diagonals negative such that the heatmap colors are red, not blu
    cm = cm * -1
    cm[np.diag_indices(cm.shape[0])] = cm.diagonal() * -1
    cm[:, cm.shape[0] - 1] = cm[:, cm.shape[0] - 1] * -1
    cm[cm.shape[0] - 1, :] = cm[cm.shape[0] - 1, :] * -1
    
    # scale the precision and recall numbers such that the colors are in the range of the diagonals
    cm[:, -1] = np.round(max(diagonal) * (cm[:, -1] / 100), 0)
    cm[-1, :] = np.round(max(diagonal) * (cm[-1, :] / 100), 0)
    cm[-1, -1] = np.round(max(diagonal) * (overall_accuracy / 100), 0)
    
    #matplotlib.rcParams.update({'font.size': font_size})
    if normalize:
        cm = cm.astype('float') / row_wise_sums[:, np.newaxis]
    else:
        pass

    labels_rec = np.concatenate((class_names,
                                 np.array(['Rec.'])))
    labels_prec = np.concatenate((class_names, 
                                  np.array(['Prec.'])))
    
    if switch_axes:
        xticklabels = labels_rec
        yticklabels = labels_prec
        ylabel = 'Predicted'
        xlabel = 'Actual'
        cm = cm.transpose()
        cm_annot = cm_annot.transpose()
    else:
        xticklabels = labels_prec
        yticklabels = labels_rec
        ylabel = 'Actual'
        xlabel = 'Predicted'
    if mask_zeros:
        mask = cm == 0
    else:
        mask = None
    
    ax = sns.heatmap(cm, 
                     vmin=vmin, 
                     vmax=vmax,
                     cmap=cmap, 
                     center=0,
                     fmt=fmt, 
                     cbar=cbar, 
                     robust=robust,
                     xticklabels=xticklabels, 
                     yticklabels=yticklabels,
                     annot=cm_annot,
                     annot_kws=annot_kws, 
                     square=True, 
                     mask=mask,
                     ax=ax)

    # fix matplotlib 3.1.1 bug
    # https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    tick_marks = np.arange(len(class_names))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    return ax
