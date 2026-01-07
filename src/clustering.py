from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def perform_clustering(data, algo='kmeans', n_clusters=5):
    """
    Executes the chosen clustering algorithm on the provided data.
    
    Args:
        data (np.array): Latent vectors from the VAE.
        algo (str): 'kmeans', 'agglomerative', or 'dbscan'.
        n_clusters (int): Number of clusters (ignored for DBSCAN).
        
    Returns:
        labels (np.array): Cluster labels for each data point.
    """
    if algo == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(data)
        
    elif algo == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(data)
        
    elif algo == 'dbscan':
        model = DBSCAN(eps=3.0, min_samples=5)
        labels = model.fit_predict(data)
        
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")
    
    return labels