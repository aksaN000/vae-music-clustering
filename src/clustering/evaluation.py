"""
Evaluation Metrics for Clustering
All 6 metrics: Silhouette, CH Index, DB Index, ARI, NMI, Purity
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional
import pandas as pd


class ClusteringMetrics:
    """Comprehensive clustering evaluation metrics"""
    
    @staticmethod
    def silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
        """
        Silhouette Score: Measures how similar an object is to its own cluster
        Range: [-1, 1], Higher is better
        """
        try:
            if len(np.unique(labels)) < 2:
                return -1.0
            score = silhouette_score(features, labels)
            return float(score)
        except:
            return -1.0
    
    @staticmethod
    def calinski_harabasz_score(features: np.ndarray, labels: np.ndarray) -> float:
        """
        Calinski-Harabasz Index (Variance Ratio Criterion)
        Ratio of between-cluster variance to within-cluster variance
        Higher is better
        """
        try:
            if len(np.unique(labels)) < 2:
                return 0.0
            score = calinski_harabasz_score(features, labels)
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def davies_bouldin_score(features: np.ndarray, labels: np.ndarray) -> float:
        """
        Davies-Bouldin Index: Average similarity of each cluster with its most similar cluster
        Lower is better
        """
        try:
            if len(np.unique(labels)) < 2:
                return float('inf')
            score = davies_bouldin_score(features, labels)
            return float(score)
        except:
            return float('inf')
    
    @staticmethod
    def adjusted_rand_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Adjusted Rand Index: Measures agreement between clustering and ground truth
        Adjusted for chance, Range: [-1, 1], Higher is better
        """
        try:
            score = adjusted_rand_score(true_labels, pred_labels)
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def normalized_mutual_info_score(
        true_labels: np.ndarray,
        pred_labels: np.ndarray
    ) -> float:
        """
        Normalized Mutual Information: Quantifies mutual information between clusters and labels
        Range: [0, 1], Higher is better
        """
        try:
            score = normalized_mutual_info_score(true_labels, pred_labels)
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def cluster_purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Cluster Purity: Fraction of dominant class in each cluster
        Range: [0, 1], Higher is better
        """
        try:
            # Create contingency matrix
            clusters = np.unique(pred_labels)
            classes = np.unique(true_labels)
            
            total = len(true_labels)
            purity_sum = 0
            
            for cluster in clusters:
                cluster_mask = pred_labels == cluster
                cluster_size = np.sum(cluster_mask)
                
                if cluster_size > 0:
                    # Find dominant class in this cluster
                    class_counts = []
                    for cls in classes:
                        count = np.sum((true_labels == cls) & cluster_mask)
                        class_counts.append(count)
                    
                    purity_sum += max(class_counts)
            
            purity = purity_sum / total
            return float(purity)
        except:
            return 0.0
    
    @staticmethod
    def cluster_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Cluster Accuracy with Hungarian algorithm for optimal label matching
        """
        try:
            # Create contingency matrix
            true_unique = np.unique(true_labels)
            pred_unique = np.unique(pred_labels)
            
            n_true = len(true_unique)
            n_pred = len(pred_unique)
            
            # Map labels to indices
            true_idx_map = {label: idx for idx, label in enumerate(true_unique)}
            pred_idx_map = {label: idx for idx, label in enumerate(pred_unique)}
            
            true_indices = np.array([true_idx_map[label] for label in true_labels])
            pred_indices = np.array([pred_idx_map[label] for label in pred_labels])
            
            # Build cost matrix
            max_dim = max(n_true, n_pred)
            cost_matrix = np.zeros((max_dim, max_dim))
            
            for i in range(n_true):
                for j in range(n_pred):
                    cost_matrix[i, j] = -np.sum((true_indices == i) & (pred_indices == j))
            
            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Calculate accuracy
            total_correct = 0
            for i, j in zip(row_ind, col_ind):
                if i < n_true and j < n_pred:
                    total_correct += -cost_matrix[i, j]
            
            accuracy = total_correct / len(true_labels)
            return float(accuracy)
        except:
            return 0.0
    
    @classmethod
    def evaluate_all(
        cls,
        features: np.ndarray,
        pred_labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics
        Args:
            features: Feature matrix
            pred_labels: Predicted cluster labels
            true_labels: Ground truth labels (optional, for supervised metrics)
        """
        metrics = {}
        
        # Internal metrics (no ground truth needed)
        metrics['silhouette_score'] = cls.silhouette_score(features, pred_labels)
        metrics['calinski_harabasz_index'] = cls.calinski_harabasz_score(features, pred_labels)
        metrics['davies_bouldin_index'] = cls.davies_bouldin_score(features, pred_labels)
        
        # External metrics (require ground truth)
        if true_labels is not None:
            metrics['adjusted_rand_index'] = cls.adjusted_rand_score(true_labels, pred_labels)
            metrics['normalized_mutual_info'] = cls.normalized_mutual_info_score(true_labels, pred_labels)
            metrics['cluster_purity'] = cls.cluster_purity(true_labels, pred_labels)
            metrics['cluster_accuracy'] = cls.cluster_accuracy(true_labels, pred_labels)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Clustering Metrics"):
        """Pretty print metrics"""
        print("\n" + "="*60)
        print(f"{title:^60}")
        print("="*60)
        
        internal_metrics = [
            'silhouette_score',
            'calinski_harabasz_index',
            'davies_bouldin_index'
        ]
        
        external_metrics = [
            'adjusted_rand_index',
            'normalized_mutual_info',
            'cluster_purity',
            'cluster_accuracy'
        ]
        
        if any(k in metrics for k in internal_metrics):
            print("\nInternal Metrics (no labels required):")
            print("-" * 60)
            
            if 'silhouette_score' in metrics:
                score = metrics['silhouette_score']
                print(f"  Silhouette Score:           {score:>10.4f}  (higher is better)")
            
            if 'calinski_harabasz_index' in metrics:
                score = metrics['calinski_harabasz_index']
                print(f"  Calinski-Harabasz Index:    {score:>10.2f}  (higher is better)")
            
            if 'davies_bouldin_index' in metrics:
                score = metrics['davies_bouldin_index']
                print(f"  Davies-Bouldin Index:       {score:>10.4f}  (lower is better)")
        
        if any(k in metrics for k in external_metrics):
            print("\nExternal Metrics (with ground truth labels):")
            print("-" * 60)
            
            if 'adjusted_rand_index' in metrics:
                score = metrics['adjusted_rand_index']
                print(f"  Adjusted Rand Index (ARI):  {score:>10.4f}  (higher is better)")
            
            if 'normalized_mutual_info' in metrics:
                score = metrics['normalized_mutual_info']
                print(f"  Normalized Mutual Info:     {score:>10.4f}  (higher is better)")
            
            if 'cluster_purity' in metrics:
                score = metrics['cluster_purity']
                print(f"  Cluster Purity:             {score:>10.4f}  (higher is better)")
            
            if 'cluster_accuracy' in metrics:
                score = metrics['cluster_accuracy']
                print(f"  Cluster Accuracy:           {score:>10.4f}  (higher is better)")
        
        print("="*60 + "\n")
    
    @staticmethod
    def save_metrics(metrics: Dict[str, float], filepath: str):
        """Save metrics to CSV"""
        df = pd.DataFrame([metrics])
        df.to_csv(filepath, index=False)
        print(f"Metrics saved to: {filepath}")


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    
    print("Testing Clustering Metrics:")
    
    # Generate test data
    X, y_true = make_blobs(n_samples=300, centers=5, random_state=42)
    
    # Cluster with K-Means
    kmeans = KMeans(n_clusters=5, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    # Evaluate
    evaluator = ClusteringMetrics()
    
    # Test individual metrics
    print("\n" + "="*60)
    print("Testing Individual Metrics:")
    print("="*60)
    
    sil = evaluator.silhouette_score(X, y_pred)
    print(f"Silhouette Score: {sil:.4f}")
    
    ch = evaluator.calinski_harabasz_score(X, y_pred)
    print(f"Calinski-Harabasz Index: {ch:.2f}")
    
    db = evaluator.davies_bouldin_score(X, y_pred)
    print(f"Davies-Bouldin Index: {db:.4f}")
    
    ari = evaluator.adjusted_rand_score(y_true, y_pred)
    print(f"Adjusted Rand Index: {ari:.4f}")
    
    nmi = evaluator.normalized_mutual_info_score(y_true, y_pred)
    print(f"Normalized Mutual Info: {nmi:.4f}")
    
    purity = evaluator.cluster_purity(y_true, y_pred)
    print(f"Cluster Purity: {purity:.4f}")
    
    acc = evaluator.cluster_accuracy(y_true, y_pred)
    print(f"Cluster Accuracy: {acc:.4f}")
    
    # Test evaluate_all
    all_metrics = evaluator.evaluate_all(X, y_pred, y_true)
    evaluator.print_metrics(all_metrics, "K-Means Clustering Results")
