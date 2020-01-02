
import numpy as np
import pandas as pd


def predict_extended(df, clf):
    """Derive probabilities, predictions, and condfidence layers.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data matris X to be predicted with ``clf``.
    clf : sklearn.Classifier
        Trained sklearn classfifier with a ``predict_proba`` method.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with the same number of rows as ``df`` and n_classes + 3 columns.
        THe columns contain the class predictions, confidence clayers (max. probability
        and the difference between the max. and second highest probability), and class
        probabilities.
    """
    def convert_to_uint8(arr):
        return arr.astype(np.uint8)
    
    
    probs = clf.predict_proba(df.values)
    pred_idx = probs.argmax(axis=1)
    pred = np.zeros_like(pred_idx).astype(np.uint8)
    for i in range(probs.shape[1]):
        pred[pred_idx == i] = clf.classes_[i]
    # get reliability layers (maximum probability and margin, i.e. maximum probability minus second highest probability)
    probs_sorted = np.sort(probs, axis=1)
    max_prob = probs_sorted[:, probs_sorted.shape[1] - 1]
    margin = (
        probs_sorted[:, probs_sorted.shape[1] - 1] - probs_sorted[:, probs_sorted.shape[1] - 2]
    )

    probs = convert_to_uint8(probs * 100)
    max_prob = convert_to_uint8(max_prob * 100)
    margin = convert_to_uint8(margin * 100)

    ndigits = len(str(max(clf.classes_)))
    prob_names = [f"prob_{cid:0{ndigits}d}" for cid in clf.classes_]
    df_result = pd.concat(
        [
            pd.DataFrame({"pred": pred, "max_prob": max_prob, "margin": margin}),
            pd.DataFrame(probs, columns=prob_names),
        ],
        axis=1,
    )
    return df_result
