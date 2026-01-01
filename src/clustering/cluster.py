"""
Clustering Algorithms
K-Means, Agglomerative, DBSCAN, GMM
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from typing import Dict, Optional, Tuple
import torch


class ClusteringAlgorithms:
    """Wrapper for various clustering algorithms"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def kmeans(
        self,
        features: np.ndarray,
        n_clusters: int,
        n_init: int = 10
    ) -> Tuple[np.ndarray, Dict]:
        """
        K-Means clustering
        Args:
            features: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters
            n_init: Number of initializations
        """
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=self.random_state,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(features)
        
        info = {
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_
        }
        
        return labels, info
    
    def agglomerative(
        self,
        features: np.ndarray,
        n_clusters: int,
        linkage: str = 'ward'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Agglomerative Hierarchical Clustering
        Args:
            features: Feature matrix
            n_clusters: Number of clusters
            linkage: 'ward', 'complete', 'average', 'single'
        """
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        labels = agg.fit_predict(features)
        
        info = {
            'n_clusters': agg.n_clusters_,
            'n_leaves': agg.n_leaves_
        }
        
        return labels, info
    
    def dbscan(
        self,
        features: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> Tuple[np.ndarray, Dict]:
        """
        DBSCAN (Density-Based Spatial Clustering)
        Args:
            features: Feature matrix
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
        """
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1
        )
        
        labels = dbscan.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        info = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'core_sample_indices': dbscan.core_sample_indices_
        }
        
        return labels, info
    
    def gmm(
        self,
        features: np.ndarray,
        n_clusters: int,
        covariance_type: str = 'full'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Gaussian Mixture Model
        Args:
            features: Feature matrix
            n_clusters: Number of components
            covariance_type: 'full', 'tied', 'diag', 'spherical'
        """
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            random_state=self.random_state,
            max_iter=100
        )
        
        gmm.fit(features)
        labels = gmm.predict(features)
        
        info = {
            'means': gmm.means_,
            'covariances': gmm.covariances_,
            'weights': gmm.weights_,
            'bic': gmm.bic(features),
            'aic': gmm.aic(features),
            'log_likelihood': gmm.score(features) * len(features)
        }
        
        return labels, info
    
    def cluster(
        self,
        features: np.ndarray,
        method: str,
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Unified clustering interface
        Args:
            features: Feature matrix
            method: 'kmeans', 'agglomerative', 'dbscan', 'gmm'
            n_clusters: Number of clusters (not needed for DBSCAN)
            **kwargs: Algorithm-specific parameters
        """
        if method == 'kmeans':
            return self.kmeans(features, n_clusters, **kwargs)
        elif method == 'agglomerative':
            return self.agglomerative(features, n_clusters, **kwargs)
        elif method == 'dbscan':
            return self.dbscan(features, **kwargs)
        elif method == 'gmm':
            return self.gmm(features, n_clusters, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}")


def extract_latent_features(model, dataloader, device):
    """
    Extract latent features from VAE model
    Args:
        model: Trained VAE model
        dataloader: DataLoader
        device: torch device
    Returns:
        features: Latent features (numpy array)
        labels: Ground truth labels (dict with 'language', 'genre', etc.)
    """
    model.eval()
    all_features = []
    all_labels = {
        'language': [],
        'genre': [],
        'id': []
    }
    
    with torch.no_grad():
        for batch in dataloader:
            # Get features based on batch structure
            if 'features' in batch:
                # Audio-only
                x = batch['features'].to(device)
                z = model.get_latent(x)
            elif 'audio_features' in batch and 'lyrics_features' in batch:
                # Multi-modal
                audio_x = batch['audio_features'].to(device)
                lyrics_x = batch['lyrics_features'].to(device)
                
                if hasattr(model, 'get_latent') and callable(getattr(model, 'get_latent')):
                    # Check if model expects both modalities
                    try:
                        z = model.get_latent(audio_x, lyrics_x)
                    except TypeError:
                        # Model only expects audio
                        z = model.get_latent(audio_x)
                else:
                    z = model.get_latent(audio_x)
            else:
                raise ValueError("Unknown batch structure")
            
            all_features.append(z.cpu().numpy())
            
            # Collect labels
            if 'language' in batch:
                all_labels['language'].extend(batch['language'].cpu().numpy())
            if 'genre' in batch:
                if isinstance(batch['genre'], torch.Tensor):
                    all_labels['genre'].extend(batch['genre'].cpu().numpy())
                else:
                    all_labels['genre'].extend(batch['genre'])
            if 'id' in batch:
                all_labels['id'].extend(batch['id'])
    
    features = np.concatenate(all_features, axis=0)
    
    # Convert to numpy arrays
    for key in all_labels:
        if all_labels[key]:
            if isinstance(all_labels[key][0], str):
                all_labels[key] = np.array(all_labels[key])
            else:
                all_labels[key] = np.array(all_labels[key])
    
    return features, all_labels


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    
    print("Testing Clustering Algorithms:")
    
    # Generate test data
    X, y_true = make_blobs(n_samples=300, centers=5, random_state=42)
    
    clustering = ClusteringAlgorithms()
    
    # Test K-Means
    print("\n1. K-Means:")
    labels, info = clustering.kmeans(X, n_clusters=5)
    print(f"   Labels: {np.unique(labels)}")
    print(f"   Inertia: {info['inertia']:.2f}")
    
    # Test Agglomerative
    print("\n2. Agglomerative:")
    labels, info = clustering.agglomerative(X, n_clusters=5)
    print(f"   Labels: {np.unique(labels)}")
    print(f"   Clusters: {info['n_clusters']}")
    
    # Test DBSCAN
    print("\n3. DBSCAN:")
    labels, info = clustering.dbscan(X, eps=0.5, min_samples=5)
    print(f"   Labels: {np.unique(labels)}")
    print(f"   Clusters found: {info['n_clusters']}")
    print(f"   Noise points: {info['n_noise']}")
    
    # Test GMM
    print("\n4. Gaussian Mixture:")
    labels, info = clustering.gmm(X, n_clusters=5)
    print(f"   Labels: {np.unique(labels)}")
    print(f"   BIC: {info['bic']:.2f}")
    print(f"   AIC: {info['aic']:.2f}")
