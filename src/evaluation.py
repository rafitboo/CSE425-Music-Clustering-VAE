from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                             davies_bouldin_score, adjusted_rand_score, 
                             normalized_mutual_info_score, confusion_matrix)
import numpy as np


from src.clustering import perform_clustering

def calculate_purity(y_true, y_pred):
    """
    Computes cluster purity: How dominant the max class is in each cluster.
    """
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def cluster_and_evaluate(data, true_labels=None, n_clusters=5, algo='kmeans'):
    """
    Runs clustering (via src.clustering) and returns metrics.
    """

    pred_labels = perform_clustering(data, algo=algo, n_clusters=n_clusters)


    if len(set(pred_labels)) > 1:
        sil_score = silhouette_score(data, pred_labels)
        ch_score = calinski_harabasz_score(data, pred_labels)
        db_score = davies_bouldin_score(data, pred_labels) 
    else:
        sil_score, ch_score, db_score = 0, 0, 0


    ari_score = 0.0
    nmi_score = 0.0
    purity_score = 0.0
    
    if true_labels is not None:
        ari_score = adjusted_rand_score(true_labels, pred_labels)
        nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
        purity_score = calculate_purity(true_labels, pred_labels)
    
    return pred_labels, sil_score, ch_score, db_score, ari_score, nmi_score, purity_score