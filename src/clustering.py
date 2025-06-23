import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import gc
from sklearn.metrics import (
    calinski_harabasz_score, silhouette_score, davies_bouldin_score
)



def evaluate_dbscan_clustering(
        embedding: np.ndarray,
        labels: np.ndarray
    ):
    """Evaluate DBSCAN clustering using Calinski-Harabasz, Silhouette, and
    Davies-Bouldin scores.

    Parameters
    ----------
    embedding : np.ndarray
        The embedding of the data points.
    labels : np.ndarray
        The cluster labels assigned by DBSCAN.

    Returns
    -------
    dict
        A dictionary containing the scores for Calinski-Harabasz, Silhouette,
        and Davies-Bouldin metrics.
    """
    if len(set(labels)) < 2:
        return {'calinski_harabasz': -1, 'silhouette': -1, 'davies_bouldin': -1}

    mask = labels != -1
    if mask.sum() < 2:
        return {'calinski_harabasz': -1, 'silhouette': -1, 'davies_bouldin': -1}

    embedding_clean = embedding[mask]
    labels_clean = labels[mask]

    try:
        ch_score = calinski_harabasz_score(embedding_clean, labels_clean)
        sil_score = silhouette_score(embedding_clean, labels_clean)
        db_score = davies_bouldin_score(embedding_clean, labels_clean)
    except:
        return {'calinski_harabasz': -1, 'silhouette': -1, 'davies_bouldin': -1}

    return {
        'calinski_harabasz': ch_score,
        'silhouette': sil_score,
        'davies_bouldin': db_score  # Raw, not inverted yet
    }


def plot_parameter_optimization(
        eps_values: list,
        min_pts_values: list,
        score_matrices: dict,
        n_clusters_matrix: np.ndarray,
        noise_percentage_matrix: np.ndarray,
        embedding_method: str
    ):
    """Plot heatmaps for different scoring metrics, number of clusters, and
    noise percentage.

    Parameters
    ----------
    eps_values : list
        List of epsilon values used in DBSCAN.
    min_pts_values : list
        List of minimum samples values used in DBSCAN.
    score_matrices : dict
        Dictionary containing matrices for Calinski-Harabasz, Silhouette,
        Davies-Bouldin, and Composite scores.
    n_clusters_matrix : np.ndarray
        Matrix containing the number of clusters for each combination of
        epsilon and minimum samples.
    noise_percentage_matrix : np.ndarray
        Matrix containing the percentage of noise points for each combination
        of epsilon and minimum samples.
    embedding_method : str
        The embedding method used (e.g., 'pca', 'tsne', 'umap').
    """
    
    # Organize all plots and their respective matrices, titles, and colormaps
    plot_data = [
        {
            "matrix": score_matrices["calinski_harabasz"],
            "title": "Calinski-Harabasz Score",
            "cmap": "bone"
        },
        {
            "matrix": score_matrices["silhouette"],
            "title": "Silhouette Score",
            "cmap": "bone"
        },
        {
            "matrix": score_matrices["davies_bouldin"],
            "title": "Davies-Bouldin Score (adj)",
            "cmap": "bone"
        },
        {
            "matrix": score_matrices["composite"],
            "title": "Composite Score",
            "cmap": "bone"
        },
        {
            "matrix": n_clusters_matrix,
            "title": "Number of Clusters",
            "cmap": "bone"
        },
        {
            "matrix": noise_percentage_matrix,
            "title": "Percentage of Noise Points",
            "cmap": "bone"
        }
    ]
    
    num_plots = len(plot_data)
    rows = int(np.ceil(num_plots / 3))
    cols = min(num_plots, 3)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = axes.ravel()

    fig.suptitle(
        f"DBSCAN Parameter Optimization ({embedding_method.upper()} Embedding)",
        fontsize=16
    )

    for idx, data in enumerate(plot_data):
        im = axes[idx].imshow(
            data["matrix"], cmap=data["cmap"], aspect="auto", origin="lower"
        )
        axes[idx].set_xlabel("eps")
        axes[idx].set_ylabel("min_samples")
        axes[idx].set_title(data["title"])

        # Set ticks
        eps_tick_indices = np.arange(
            0, len(eps_values), max(1, len(eps_values) // 5)
        )
        min_pts_tick_indices = np.arange(
            0, len(min_pts_values), max(1, len(min_pts_values) // 5)
        )

        axes[idx].set_xticks(eps_tick_indices)
        axes[idx].set_xticklabels(
            [f"{eps_values[i]:.1f}" for i in eps_tick_indices]
        )
        axes[idx].set_yticks(min_pts_tick_indices)
        axes[idx].set_yticklabels(
            [f"{min_pts_values[i]}" for i in min_pts_tick_indices]
        )

        plt.colorbar(im, ax=axes[idx])

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def run_dbscan_opt(
        adata: sc.AnnData,
        n_neighbors: int,
        elbow: int,
        min_pts_values: list,
        eps_values: list,
        noise_threshold_penalize: int,
        noise_threshold_discard: int,
        seed: int
    ):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=elbow)
    sc.tl.tsne(adata, random_state=seed)
    sc.tl.umap(adata, random_state=seed)

    embedding_results = {}

    for embedding_method in ["pca", "tsne", "umap"]:
        print(
            f"\nRunning DBSCAN optimization with {embedding_method} "
            "embeddings..."
        )

        if embedding_method == "pca":
            embedding = adata.obsm["X_pca"][:, :elbow]
        elif embedding_method == "tsne":
            embedding = adata.obsm["X_tsne"]
        elif embedding_method == "umap":
            embedding = adata.obsm["X_umap"]

        score_matrices = {
            "calinski_harabasz": np.zeros(
                (len(min_pts_values), len(eps_values))
            ),
            "silhouette": np.zeros(
                (len(min_pts_values), len(eps_values))
            ),
            "davies_bouldin": np.zeros(
                (len(min_pts_values), len(eps_values))
            ),
            "composite": np.zeros(
                (len(min_pts_values), len(eps_values))
            )
        }

        n_clusters_matrix = np.zeros((len(min_pts_values), len(eps_values)))
        n_noise_matrix = np.zeros((len(min_pts_values), len(eps_values)))
        noise_percentage_matrix = np.zeros(
            (len(min_pts_values), len(eps_values))
        )

        for i, min_pts in tqdm(
            enumerate(min_pts_values),
            desc=f"Optimizing {embedding_method.upper()}",
            total=len(min_pts_values),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
        ):
            for j, eps in enumerate(eps_values):
                db = DBSCAN(eps=eps, min_samples=min_pts).fit(embedding)
                labels = db.labels_

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_percentage = (n_noise / len(labels)) * 100

                n_clusters_matrix[i, j] = n_clusters
                n_noise_matrix[i, j] = n_noise
                noise_percentage_matrix[i, j] = noise_percentage

                scores = evaluate_dbscan_clustering(embedding, labels)
                for metric in [
                    "calinski_harabasz", "silhouette", "davies_bouldin"
                ]:
                    score_matrices[metric][i, j] = scores[metric]

                # Garbage collection to manage memory usage
                del db
                del labels
                del n_clusters
                del n_noise
                del scores
                gc.collect()
            gc.collect()

        # Dynamic normalization after grid search:
        norm_score_matrices = {}
        for metric in ["calinski_harabasz", "silhouette", "davies_bouldin"]:
            matrix = score_matrices[metric]
            if metric == "davies_bouldin":
                matrix = -matrix  # Invert so higher is better
            min_val = np.min(matrix)
            max_val = np.max(matrix)
            if max_val - min_val == 0:
                norm_matrix = np.zeros_like(matrix)
            else:
                norm_matrix = (matrix - min_val) / (max_val - min_val)
            norm_score_matrices[metric] = norm_matrix

        # Composite with noise penalty
        noise_penalty = np.maximum(
            0, (noise_percentage_matrix - noise_threshold_penalize) / 100
        )
        composite_matrix = (
            0.33 * norm_score_matrices["calinski_harabasz"] +
            0.33 * norm_score_matrices["silhouette"] +
            0.33 * norm_score_matrices["davies_bouldin"] -
            noise_penalty
        )
        score_matrices["composite"] = composite_matrix

        # Discard configurations with excessive noise
        composite_matrix[noise_percentage_matrix > noise_threshold_discard] = -1

        # Plot normalized metrics and composite, and the new cluster/noise plots
        plot_parameter_optimization(
            eps_values, min_pts_values,
            score_matrices,
            n_clusters_matrix,
            noise_percentage_matrix,
            embedding_method
        )

        print(
            f"\nOptimal parameters for {embedding_method.upper()} "
            "based on composite score:"
        )
        best_idx = np.unravel_index(
            np.argmax(score_matrices["composite"]), composite_matrix.shape
        )
        best_min_pts = min_pts_values[best_idx[0]]
        best_eps = eps_values[best_idx[1]]
        best_score = composite_matrix[best_idx[0], best_idx[1]]
        best_n_clusters = n_clusters_matrix[best_idx[0], best_idx[1]]
        best_n_noise = n_noise_matrix[best_idx[0], best_idx[1]]
        best_noise_pct = noise_percentage_matrix[best_idx[0], best_idx[1]]

        gc.collect()

        print(f"  eps: {best_eps:.2f}")
        print(f"  min_samples: {best_min_pts}")
        print(f"  Composite Score: {best_score:.3f}")
        print(f"  Number of clusters: {int(best_n_clusters)}")
        print(f"  Noise points: {int(best_n_noise)} ({best_noise_pct:.1f}%)")

        embedding_results[embedding_method] = {
            "eps": best_eps,
            "min_samples": best_min_pts,
            "composite_score": best_score,
            "n_clusters": best_n_clusters,
            "noise_percentage": best_noise_pct
        }

        # Apply optimal clustering and label cells
        print(f"\nApplying optimal DBSCAN clustering for {embedding_method}...")
        db_optimal = DBSCAN(
            eps=best_eps, min_samples=best_min_pts
        ).fit(embedding)
        adata.obs[
            f"dbscan_{embedding_method}"
        ] = db_optimal.labels_.astype(str)
        adata.obs[
            f"dbscan_{embedding_method}"
        ] = adata.obs[f"dbscan_{embedding_method}"].replace("-1", "Noise")

        final_noise_pct = (
            list(db_optimal.labels_).count(-1) / len(db_optimal.labels_)
        ) * 100
        if final_noise_pct > 50:
            print(
                f"WARN: {final_noise_pct:.1f}% of cells classified as noise. "
                "Consider:"
            )
            print("  - Adjusting parameter ranges")
            print("  - Using different preprocessing")
            print("  - This embedding may not be suitable for DBSCAN")

        if hasattr(adata, 'obsm') and f"X_umap" in adata.obsm:
            sc.pl.umap(
                adata,
                color=[f"dbscan_{embedding_method}"],
                title=f"DBSCAN on {embedding_method.upper()} "
                "(eps={best_eps:.2f}, min_samples={best_min_pts})",
                show=True
            )
        else:
            print(
                f"Skipping UMAP plot for {embedding_method} as X_umap is not "
                "available in adata.obsm."
            )

        gc.collect()

    print("\n" + "="*60)
    print("FINAL SUMMARY AND RECOMMENDATIONS:")
    print("="*60)
    print("\nEmbedding Comparison:")
    print(
        f"{'Method':<8} {'Composite Score':<15} {'Clusters':<10} "
        f"{'Noise %':<8} {'eps':<6} {'min_samples':<12}"
    )
    print("-" * 70)

    best_method = None
    best_composite_score = -np.inf

    for method, results in embedding_results.items():
        print(
            f"{method.upper():<8} {results['composite_score']:<15.3f} "
            f"{results['n_clusters']:<10} {results['noise_percentage']:<8.1f} "
            f"{results['eps']:<6.2f} {results['min_samples']:<12}"
        )
        if results['composite_score'] > best_composite_score:
            best_composite_score = results['composite_score']
            best_method = method

    print(
        f"\nRECOMMENDED: {best_method.upper()} embedding with composite score "
        f"{best_composite_score:.3f}"
    )

    print("\nGuidelines:")
    print("1. Choose the embedding with the highest composite score")
    print("2. Ensure noise percentage is reasonable (typically < 30-40%)")
    print("3. Visually inspect the clustering results")
    print("4. Validate with biological knowledge of your cell types")
    print("5. Consider the number of clusters relative to expected cell types")

    print("\nOptimization complete.")
    gc.collect()