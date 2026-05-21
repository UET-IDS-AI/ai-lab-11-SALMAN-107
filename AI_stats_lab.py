"""
AI_stats_lab.py

Lab: Unsupervised Learning and K-Means Clustering

Topics:
- Unsupervised learning with unlabeled data
- Iris dataset without labels
- Feature standardization
- K-Means clustering
- K-Means objective function
- Elbow method for choosing K
- Underfitting and overfitting in clustering
- Distance-based outlier detection
- Visualization of unlabeled data, clusters, centroids, and elbow curve

Instructions:
- Implement all functions.
- Do NOT change function names.
- Do NOT print inside functions.
- Return exactly the required formats.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# ============================================================
# Question 1: Unlabeled Data and K-Means Clustering
# ============================================================

def load_iris_unlabeled(feature_indices=(0, 1)):
    """
    Load the Iris dataset without labels.

    Parameters:
        feature_indices : tuple
            Indices of features to use.
            Default (0, 1) means:
                sepal length
                sepal width

    Returns:
        A dictionary:

        {
            "X": feature matrix with selected columns,
            "feature_names": list of selected feature names
        }

    Notes:
        - Do NOT return the class labels.
        - This is an unsupervised learning setup.
    """
    iris = load_iris()
    X = iris.data[:, feature_indices]
    feature_names = [iris.feature_names[i] for i in feature_indices]
    
    return {
        "X": X,
        "feature_names": feature_names
    }


def standardize_features(X):
    """
    Standardize features to zero mean and unit variance.

    Parameters:
        X : NumPy array of shape (n_samples, n_features)

    Returns:
        A dictionary:

        {
            "X_scaled": standardized feature matrix,
            "mean": feature-wise mean,
            "std": feature-wise standard deviation
        }

    Formula:
        X_scaled = (X - mean) / std

    Notes:
        - If any std value is 0, replace it with 1 before division.
        - This avoids division by zero.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Replace zero std with 1 to avoid division by zero
    std = np.where(std == 0, 1, std)
    
    X_scaled = (X - mean) / std
    
    return {
        "X_scaled": X_scaled,
        "mean": mean,
        "std": std
    }


def fit_kmeans(X, K, random_state=0, n_init=10):
    """
    Fit K-Means clustering on data X.

    Parameters:
        X            : feature matrix
        K            : number of clusters
        random_state : random seed
        n_init       : number of centroid initializations

    Returns:
        A dictionary:

        {
            "centroids": learned centroids,
            "labels": cluster assignment for each point,
            "objective": K-Means objective value,
            "n_iter": number of iterations used
        }

    Notes:
        - Use sklearn.cluster.KMeans.
        - The K-Means objective is the sum of squared distances
          from each point to its assigned centroid.
        - In sklearn, this value is stored in model.inertia_.
    """
    model = KMeans(n_clusters=K, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    
    return {
        "centroids": model.cluster_centers_,
        "labels": labels,
        "objective": model.inertia_,
        "n_iter": model.n_iter_
    }


def compute_kmeans_objective(X, centroids, labels):
    """
    Compute the K-Means objective manually.

    Parameters:
        X         : feature matrix
        centroids : centroid matrix of shape (K, n_features)
        labels    : assigned cluster index for each point

    Returns:
        objective : sum of squared distances from each point
                    to its assigned centroid

    Formula:
        objective = sum_i || x_i - c_{label_i} ||^2
    """
    objective = 0.0
    for i in range(len(X)):
        centroid = centroids[labels[i]]
        distance_squared = np.sum((X[i] - centroid) ** 2)
        objective += distance_squared
    
    return objective


# ============================================================
# Question 2: Choosing K, Underfitting/Overfitting, and Outliers
# ============================================================

def evaluate_k_values(X, k_values, random_state=0, n_init=10):
    """
    Run K-Means for multiple values of K.

    Parameters:
        X            : feature matrix
        k_values     : list of K values
        random_state : random seed
        n_init       : number of centroid initializations

    Returns:
        A dictionary:

        {
            "k_values": list of K values,
            "objectives": list of objective values,
            "relative_improvements": list of relative improvements
        }

    Relative improvement:
        For the first K, improvement is 0.0.
        For later K values:

        improvement = (previous_objective - current_objective) / previous_objective

    Notes:
        - Objective should usually decrease as K increases.
        - Very large K can overfit by creating too many small clusters.
    """
    objectives = []
    relative_improvements = []
    
    for k in k_values:
        result = fit_kmeans(X, k, random_state=random_state, n_init=n_init)
        objectives.append(result["objective"])
    
    # Compute relative improvements
    for i, obj in enumerate(objectives):
        if i == 0:
            relative_improvements.append(0.0)
        else:
            prev_obj = objectives[i - 1]
            improvement = (prev_obj - obj) / prev_obj
            relative_improvements.append(improvement)
    
    return {
        "k_values": k_values,
        "objectives": objectives,
        "relative_improvements": relative_improvements
    }


def choose_elbow_k(k_values, objectives):
    """
    Choose K using a simple elbow heuristic.

    Parameters:
        k_values   : list of K values
        objectives : list of K-Means objective values

    Returns:
        best_k : K value at the elbow point

    Method:
        Use the maximum-distance-to-line heuristic.

        1. Treat the first and last points of the objective curve
           as endpoints of a straight line.
        2. Compute the perpendicular distance of each intermediate point
           from this line.
        3. Return the K value with the largest distance.

    Notes:
        - If fewer than 3 K values are given, return the first K.
        - This is a heuristic, not a perfect rule.
    """
    if len(k_values) < 3:
        return k_values[0]
    
    # Convert to numpy arrays for easier computation
    k_vals = np.array(k_values, dtype=float)
    obj_vals = np.array(objectives, dtype=float)
    
    # First and last points
    point1 = np.array([k_vals[0], obj_vals[0]])
    point2 = np.array([k_vals[-1], obj_vals[-1]])
    
    # Vector from point1 to point2
    line_vec = point2 - point1
    line_len_sq = np.dot(line_vec, line_vec)
    
    max_distance = -1
    best_k = k_values[0]
    
    # Check each intermediate point
    for i in range(1, len(k_values) - 1):
        point = np.array([k_vals[i], obj_vals[i]])
        
        # Vector from point1 to the point
        point_vec = point - point1
        
        # Project point_vec onto line_vec
        t = np.dot(point_vec, line_vec) / line_len_sq
        
        # Find the closest point on the line
        closest_point = point1 + t * line_vec
        
        # Compute perpendicular distance
        distance = np.linalg.norm(point - closest_point)
        
        if distance > max_distance:
            max_distance = distance
            best_k = k_values[i]
    
    return best_k


def cluster_size_summary(labels, K):
    """
    Count how many data points belong to each cluster.

    Parameters:
        labels : cluster assignment for each point
        K      : number of clusters

    Returns:
        A dictionary:

        {
            cluster_index: number_of_points_in_that_cluster
        }

    Example:
        labels = [0, 0, 1, 1, 1]
        K = 2

        output:
        {
            0: 2,
            1: 3
        }
    """
    summary = {}
    for k in range(K):
        count = np.sum(labels == k)
        summary[k] = count
    
    return summary


def identify_outliers_by_distance(X, centroids, labels, top_n=5):
    """
    Identify possible outliers based on distance from assigned centroid.

    Parameters:
        X         : feature matrix
        centroids : centroid matrix
        labels    : assigned cluster label for each point
        top_n     : number of farthest points to return

    Returns:
        A dictionary:

        {
            "indices": indices of top_n farthest points,
            "distances": squared distances of those points
        }

    Notes:
        - A point far from its assigned centroid may be unusual.
        - Sort points by distance in descending order.
        - Return the top_n farthest points.
    """
    distances_sq = []
    
    for i in range(len(X)):
        centroid = centroids[labels[i]]
        dist_sq = np.sum((X[i] - centroid) ** 2)
        distances_sq.append(dist_sq)
    
    distances_sq = np.array(distances_sq)
    
    # Get indices of top_n largest distances
    top_indices = np.argsort(distances_sq)[::-1][:top_n]
    top_distances = distances_sq[top_indices]
    
    return {
        "indices": top_indices.tolist(),
        "distances": top_distances.tolist()
    }


def diagnose_clustering_fit(K, elbow_k):
    """
    Diagnose whether the chosen K is likely underfitting, good fit, or overfitting.

    Parameters:
        K        : chosen number of clusters
        elbow_k  : elbow-method recommended K

    Returns:
        diagnosis string:

        if K < elbow_k:
            "underfitting"

        if K == elbow_k:
            "good_fit"

        if K > elbow_k:
            "overfitting"

    Notes:
        - In clustering, very small K may merge true groups.
        - Very large K may split meaningful groups into tiny clusters.
    """
    if K < elbow_k:
        return "underfitting"
    elif K == elbow_k:
        return "good_fit"
    else:
        return "overfitting"


# ============================================================
# Question 3: Visualization
# ============================================================

def plot_unlabeled_data(X, feature_names=None, title="Unlabeled Data"):
    """
    Visualize unlabeled 2D data.

    Parameters:
        X             : feature matrix of shape (n_samples, 2)
        feature_names : optional list of two feature names
        title         : plot title

    Returns:
        fig, ax

    Notes:
        - This function should create a scatter plot.
        - Do not use labels/classes because this is unsupervised learning.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50)
    
    ax.set_title(title)
    
    if feature_names is not None:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    
    return fig, ax


def plot_kmeans_clusters(X, labels, centroids, feature_names=None, title="K-Means Clusters"):
    """
    Visualize K-Means clustering results.

    Parameters:
        X             : feature matrix of shape (n_samples, 2)
        labels        : cluster assignment for each point
        centroids     : learned cluster centroids
        feature_names : optional list of two feature names
        title         : plot title

    Returns:
        fig, ax

    Notes:
        - Plot data points colored by cluster label.
        - Plot centroids using a large marker.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data points colored by cluster
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Centroids')
    
    ax.set_title(title)
    
    if feature_names is not None:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    
    return fig, ax


def plot_elbow_curve(k_values, objectives, title="Elbow Method"):
    """
    Plot K-Means objective values versus K.

    Parameters:
        k_values   : list of K values
        objectives : list of objective values
        title      : plot title

    Returns:
        fig, ax

    Notes:
        - X-axis should show K.
        - Y-axis should show objective value.
        - This plot helps identify the elbow point.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(k_values, objectives, 'b-o', linewidth=2, markersize=8)
    
    ax.set_title(title)
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Objective value")
    ax.grid(True, alpha=0.3)
    
    return fig, ax


if __name__ == "__main__":
    print("Implement all required functions.")
