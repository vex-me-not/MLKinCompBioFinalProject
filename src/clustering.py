import anndata as ad
import scanpy as sc
import scanorama

import numpy as np
import pandas as pd
import random

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from tqdm import tqdm
import gc

from sklearn.metrics import (
    calinski_harabasz_score, silhouette_score, davies_bouldin_score
)

from src.utils import find_elbow_pcs

from typing import Optional

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
        return {"calinski_harabasz": -1, "silhouette": -1, "davies_bouldin": -1}

    mask = labels != -1
    if mask.sum() < 2:
        return {"calinski_harabasz": -1, "silhouette": -1, "davies_bouldin": -1}

    embedding_clean = embedding[mask]
    labels_clean = labels[mask]

    try:
        ch_score = calinski_harabasz_score(embedding_clean, labels_clean)
        sil_score = silhouette_score(embedding_clean, labels_clean)
        db_score = davies_bouldin_score(embedding_clean, labels_clean)
    except:
        return {"calinski_harabasz": -1, "silhouette": -1, "davies_bouldin": -1}

    return {
        "calinski_harabasz": ch_score,
        "silhouette": sil_score,
        "davies_bouldin": db_score  # Raw, not inverted yet
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
    """Run DBSCAN optimization on the provided AnnData object.
    
    Parameters
    ----------
    adata : sc.AnnData
        The annotated data object containing single-cell data.
    n_neighbors : int
        Number of neighbors to use for nearest neighbor graph.
    elbow : int
        Number of principal components to use for PCA.
    min_pts_values : list
        List of minimum samples values to test for DBSCAN.
    eps_values : list
        List of epsilon values to test for DBSCAN.
    noise_threshold_penalize : int
        Percentage of noise points above which to apply a penalty in the
        composite score.
    noise_threshold_discard : int
        Percentage of noise points above which to discard the configuration
        entirely.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    None
        The function modifies the `adata` object in place, adding DBSCAN
        clustering results to `adata.obs`.
    """
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

        if hasattr(adata, "obsm") and f"X_umap" in adata.obsm:
            sc.pl.umap(
                adata,
                color=[f"dbscan_{embedding_method}"],
                title=f"DBSCAN on {embedding_method.upper()} (eps={best_eps:.2f}, min_samples={best_min_pts})",
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
        f"{"Method":<8} {"Composite Score":<15} {"Clusters":<10} "
        f"{"Noise %":<8} {"eps":<6} {"min_samples":<12}"
    )
    print("-" * 70)

    best_method = None
    best_composite_score = -np.inf

    for method, results in embedding_results.items():
        print(
            f"{method.upper():<8} {results["composite_score"]:<15.3f} "
            f"{results["n_clusters"]:<10} {results["noise_percentage"]:<8.1f} "
            f"{results["eps"]:<6.2f} {results["min_samples"]:<12}"
        )
        if results["composite_score"] > best_composite_score:
            best_composite_score = results["composite_score"]
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

    del embedding_results
    del score_matrices
    del n_clusters_matrix
    del n_noise_matrix
    del noise_percentage_matrix
    del embedding
    del db_optimal

    gc.collect()


def plot_dendro_heatmap(
        adata,
        Z,
        cluster_labels_key,
        dendrogram_title,
        marker_genes,
        color_clusters=None  # NEW: Number of clusters to color in dendrogram
    ):
    """
    Recreates a figure similar to the first example: a horizontal dendrogram
    aligned with heatmaps for annotations and marker gene expression.

    Parameters
    ----------
    adata : sc.AnnData)
        Annotated data object.
    Z : np.ndarray
        The linkage matrix from hierarchical clustering.
    cluster_labels_key : str
        The key in adata.obs for the cluster labels.
    dendrogram_title : str
        Title for the plot.
    marker_genes : list
        A list of marker genes to plot in the heatmap.
    color_clusters : int, optional
        Number of clusters to color in the dendrogram. If None, uses default
        coloring.
    """
    print("Generating dendrogram with aligned heatmaps...")
    
    # Debug: Check marker genes
    print(f"Number of marker genes provided: {len(marker_genes)}")
    print(f"First 10 marker genes: {marker_genes[:10]}")
    
    # Filter marker genes to only those present in the data
    available_genes = list(adata.var_names)
    valid_marker_genes = [
        gene for gene in marker_genes if gene in available_genes
    ]
    
    print(
        f"Number of valid marker genes found in data: {len(valid_marker_genes)}"
    )

    if len(valid_marker_genes) == 0:
        print("ERROR: No marker genes found in the data!")
        print(f"Available genes in data: {available_genes[:20]}...")
        print("Skipping gene expression heatmap...")
        return
    
    # Limit to top genes if too many
    if len(valid_marker_genes) > 50:
        valid_marker_genes = valid_marker_genes[:50]
        print(f"Limited to top 50 marker genes for visualization")
    
    # Get the order of cells from the dendrogram
    with plt.rc_context({"lines.linewidth": 0.5}):
        dendro = sch.dendrogram(Z, no_plot=True)
    dendro_order = dendro["leaves"]
    
    # Reorder adata based on the dendrogram
    adata_ordered = adata[dendro_order, :].copy()

    # --- Create the figure layout ---
    # We use GridSpec for precise control over subplot placement.
    # We'll have one wide column for the dendrogram and several narrow columns
    # for heatmaps.
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.5, 0.2, 0.2, 2])
    
    # --- Dendrogram Panel ---
    ax_dendro = fig.add_subplot(gs[0])
    
    # Calculate color threshold if color_clusters is specified
    color_threshold = None
    if (
        color_clusters is not None
    ) and (
        color_clusters > 1
    ) and (
        color_clusters <= len(Z) + 1
    ):
        color_threshold = Z[-(color_clusters-1), 2]
        print(
            f"Coloring {color_clusters} clusters "
            f"with threshold: {color_threshold:.3f}"
        )
    
    with plt.rc_context({"lines.linewidth": 0.7}):
        dendro_kwargs = {
            "Z": Z,
            "ax": ax_dendro,
            "orientation": "left",
            "labels": adata_ordered.obs.index,
            "distance_sort": "descending",
            "show_leaf_counts": False,
            "leaf_font_size": 0  # Hide leaf labels
        }
        
        # Add color threshold if specified
        if color_threshold is not None:
            dendro_kwargs["color_threshold"] = color_threshold
            dendro_kwargs["above_threshold_color"] = "lightgray"
        
        sch.dendrogram(**dendro_kwargs)
    
    ax_dendro.spines["top"].set_visible(False)
    ax_dendro.spines["right"].set_visible(False)
    ax_dendro.spines["bottom"].set_visible(False)
    ax_dendro.spines["left"].set_visible(False)
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    
    # Update title to include cluster coloring info
    title = dendrogram_title
    if color_clusters is not None:
        title += f" ({color_clusters} clusters colored)"
    ax_dendro.set_title(title, loc="center")

    # --- Annotation Heatmap 1: Cluster Labels ---
    ax_heatmap_clusters = fig.add_subplot(gs[1])
    cluster_colors = adata_ordered.obs[
        cluster_labels_key
    ].astype("category").cat.codes
    # Use a colormap with enough distinct colors
    cmap_clusters = plt.get_cmap("Paired", len(np.unique(cluster_colors)))
    sns.heatmap(
        cluster_colors.to_frame(),
        ax=ax_heatmap_clusters,
        cmap=cmap_clusters,
        cbar=False,
        yticklabels=False
    )
    ax_heatmap_clusters.set_xticklabels(["Cluster"], rotation=90)

    # --- Annotation Heatmap 2: Batch/Sample Origin ---
    ax_heatmap_batch = fig.add_subplot(gs[2])
    batch_colors = adata_ordered.obs["batch"].astype("category").cat.codes
    cmap_batch = plt.get_cmap("coolwarm", len(np.unique(batch_colors)))
    sns.heatmap(
        batch_colors.to_frame(),
        ax=ax_heatmap_batch,
        cmap=cmap_batch,
        cbar=False,
        yticklabels=False
    )
    ax_heatmap_batch.set_xticklabels(["Batch"], rotation=90)

    # --- Gene Expression Heatmap ---
    ax_heatmap_genes = fig.add_subplot(gs[3])
    
    try:
        # Get expression data for marker genes. Using .raw to get non-scaled
        # data if available.
        if adata.raw is not None:
            expr_data = adata_ordered.raw[:, valid_marker_genes].X
            if hasattr(expr_data, "toarray"):
                expr_data = expr_data.toarray()
        else:
            expr_data = adata_ordered[:, valid_marker_genes].X
            if hasattr(expr_data, "toarray"):
                expr_data = expr_data.toarray()
        
        print(f"Expression data shape: {expr_data.shape}")
        
        # Check if we have valid expression data
        if expr_data.size == 0:
            print("ERROR: Expression data is empty!")
            return
        
        # Ensure expr_data is 2D
        if expr_data.ndim == 1:
            expr_data = expr_data.reshape(-1, 1)
        
        # Standardize each gene's expression for visualization
        # Use with_mean=True for proper standardization
        expr_data_scaled = StandardScaler().fit_transform(expr_data)
        
        print(f"Scaled expression data shape: {expr_data_scaled.shape}")
        
        sns.heatmap(
            expr_data_scaled,
            ax=ax_heatmap_genes,
            cmap="bwr",
            yticklabels=False,
            cbar_kws={"label": "Scaled Expression"}
        )
        ax_heatmap_genes.set_xticks(np.arange(len(valid_marker_genes)) + 0.5)
        ax_heatmap_genes.set_xticklabels(
            valid_marker_genes, rotation=90, ha="center"
        )
        
    except Exception as e:
        print(f"Error creating gene expression heatmap: {e}")
        print("Skipping gene expression heatmap...")
        # Remove the gene heatmap subplot and adjust layout
        ax_heatmap_genes.remove()

    plt.tight_layout(pad=0.5, w_pad=0.5)
    plt.show()


# Alternative helper function for easier cluster coloring
def plot_dendro_heatmap_with_clusters(
        adata,
        Z,
        cluster_labels_key,
        dendrogram_title,
        marker_genes,
        num_clusters
    ):
    """
    Wrapper function with simpler interface for cluster coloring.
    
    Parameters
    ----------
    dendro_clusters : int
        Number of clusters to color in the dendrogram (default: 3)
    """
    return plot_dendro_heatmap(
        adata=adata,
        Z=Z,
        cluster_labels_key=cluster_labels_key,
        dendrogram_title=dendrogram_title,
        marker_genes=marker_genes,
        color_clusters=num_clusters
    )


def plot_combined_clustermap(
    adata,
    Z_cells,
    cluster_labels_key,
    batch_key,
    plot_title,
    marker_genes,
    num_clusters_for_coloring
):
    """
    Generates the final, polished combined clustermap.

    This version includes:
    - Coloring on both cell and gene dendrograms.
    - Correctly positioned annotation labels.
    - Horizontal AND vertical dashed lines to separate clusters.
    """
    print("Generating final polished clustermap with cluster separators...")

    # --- 1. Validate and filter marker genes ---
    available_genes = list(adata.var_names)
    valid_marker_genes = [g for g in marker_genes if g in available_genes]
    if not valid_marker_genes:
        print("ERROR: No valid marker genes found in the data. Skipping plot.")
        return
    
    if len(valid_marker_genes) > 100:
        print(f"Warning: Limiting to 100 genes for visualization.")
        valid_marker_genes = valid_marker_genes[:100]

    # --- 2. Order cells based on the provided cell dendrogram (Z_cells) ---
    with plt.rc_context({"lines.linewidth": 0.5}):
        dendro_cells = sch.dendrogram(Z_cells, no_plot=True)
    cell_order_indices = dendro_cells["leaves"]
    adata_ordered_cells = adata[cell_order_indices, :].copy()

    # --- 3. Cluster genes based on their expression patterns ---
    expr_data = adata_ordered_cells[:, valid_marker_genes].X
    if hasattr(expr_data, "toarray"):
        expr_data = expr_data.toarray()
    
    expr_data_scaled = StandardScaler().fit_transform(expr_data)
    Z_genes = sch.linkage(expr_data_scaled.T, method="ward", metric="euclidean")
    with plt.rc_context({"lines.linewidth": 0.5}):
        dendro_genes = sch.dendrogram(Z_genes, no_plot=True)
    gene_order_indices = dendro_genes["leaves"]
    
    genes_ordered = [valid_marker_genes[i] for i in gene_order_indices]
    expr_data_final_ordered = expr_data_scaled[:, gene_order_indices]

    # --- 4. Create the plot layout using GridSpec ---
    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(
        2, 4,
        width_ratios=[1.5, 0.2, 0.2, 6],
        height_ratios=[1, 6],
        wspace=0.0, hspace=0.0 
    )

    ax_cell_dendro = fig.add_subplot(gs[1, 0])
    ax_annot_cluster = fig.add_subplot(gs[1, 1])
    ax_annot_batch = fig.add_subplot(gs[1, 2])
    ax_heatmap = fig.add_subplot(gs[1, 3])
    ax_gene_dendro = fig.add_subplot(gs[0, 3])
    cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.5])

    # --- 5. Plot each component ---
    # (a) Cell Dendrogram (Left)
    color_threshold_cells = Z_cells[-(num_clusters_for_coloring-1), 2]
    with plt.rc_context({"lines.linewidth": 0.8}):
        sch.dendrogram(
            Z_cells, ax=ax_cell_dendro, orientation="left",
            distance_sort="descending", color_threshold=color_threshold_cells,
            above_threshold_color="lightgray"
        )
    ax_cell_dendro.spines[:].set_visible(False)
    ax_cell_dendro.set_xticks([])
    ax_cell_dendro.set_yticks([])
    fig.suptitle(plot_title, fontsize=16, y=0.92)

    # (b) Cell Annotation Heatmaps (Middle)
    cluster_codes = adata_ordered_cells.obs[
        cluster_labels_key
    ].astype("category").cat.codes
    cmap_clusters = plt.get_cmap("Paired", len(cluster_codes.unique()))
    sns.heatmap(
        cluster_codes.to_frame(), ax=ax_annot_cluster, cmap=cmap_clusters,
        cbar=False, xticklabels=True, yticklabels=False
    )
    ax_annot_cluster.set_xticklabels(["Cluster"], rotation=90)
    ax_annot_cluster.set_ylabel("")
    for label in ax_annot_cluster.get_xticklabels():
        label.set_verticalalignment("top")

    batch_codes = adata_ordered_cells.obs[
        batch_key
    ].astype("category").cat.codes
    cmap_batch = plt.get_cmap("PiYG", len(batch_codes.unique()))
    sns.heatmap(
        batch_codes.to_frame(), ax=ax_annot_batch, cmap=cmap_batch,
        cbar=False, xticklabels=True, yticklabels=False
    )
    ax_annot_batch.set_xticklabels(["Batch"], rotation=90)
    ax_annot_batch.set_ylabel("")
    for label in ax_annot_batch.get_xticklabels():
        label.set_verticalalignment("top")

    # (c) Gene Dendrogram (Top)
    color_threshold_genes = Z_genes[-(num_clusters_for_coloring - 1), 2]
    with plt.rc_context({"lines.linewidth": 0.8}):
        sch.dendrogram(
            Z_genes, ax=ax_gene_dendro, orientation="top",
            distance_sort="descending",
            color_threshold=color_threshold_genes,
            above_threshold_color="lightgray"
        )
    ax_gene_dendro.spines[:].set_visible(False)
    ax_gene_dendro.set_xticks([])
    ax_gene_dendro.set_yticks([])

    # (d) Main Expression Heatmap (Center)
    sns.heatmap(
        expr_data_final_ordered, ax=ax_heatmap, cmap="bwr",
        yticklabels=False, xticklabels=genes_ordered,
        cbar_ax=cbar_ax, cbar_kws={"label": "Scaled Expression"}
    )
    ax_heatmap.set_xticklabels(
        ax_heatmap.get_xticklabels(), rotation=90, ha="center"
    )
    ax_heatmap.set_ylabel("")
    plt.setp(ax_heatmap.get_xticklabels(), fontsize=8)

    # (e) Add separator lines for cell and gene clusters
    # Add horizontal lines for cell clusters
    cell_clusters_ordered_vals = adata_ordered_cells.obs[
        cluster_labels_key
    ].cat.codes.values
    cluster_boundaries = np.where(
        cell_clusters_ordered_vals[:-1] != cell_clusters_ordered_vals[1:]
    )[0] + 1
    for boundary in cluster_boundaries:
        ax_heatmap.axhline(
            y=boundary, color="black", linewidth=1.0, linestyle="--"
        )

    # NEW: Add vertical lines for gene clusters
    num_gene_clusters = num_clusters_for_coloring
    gene_cluster_labels = sch.fcluster(
        Z_genes, t=num_gene_clusters, criterion="maxclust"
    )
    ordered_gene_cluster_labels = gene_cluster_labels[gene_order_indices]
    gene_boundaries = np.where(
        ordered_gene_cluster_labels[:-1] != ordered_gene_cluster_labels[1:]
    )[0] + 1
    for boundary in gene_boundaries:
        ax_heatmap.axvline(
            x=boundary, color="black", linewidth=1.0, linestyle="--"
        )

    fig.tight_layout(rect=[0, 0, 0.92, 0.9])
    plt.show()


def hierarchical_clustering(
        batches: list[sc.AnnData],
        batch_keys: list[str],
        num_clusters: int,
        min_pts: int,
        eps: float,
        n_pcs: Optional[int] = None,
        m : Optional[int] = None,
        n : Optional[int] = None,
        embedding_method: str = "PCA",
        n_neighbors: int = 15,
        scale_max_val : int = 10,
        dendrogram_limit: int = 50,
        clustermap_limit: Optional[int] = None,
        markers_per_clust: int = 5,
        seed : int = 42,
    ):
    """
    Perform hierarchical clustering on single-cell data with batch correction,
    PCA, nearest neighbors, noise detection using DBSCAN, and visualization.

    Parameters
    ----------
    batches : list[sc.AnnData]
        List of AnnData objects, each representing a batch of single-cell data.
    batch_keys : list[str]
        List of keys corresponding to each batch in the batches list.
    num_clusters : int
        Number of clusters to extract from the hierarchical clustering.
    min_pts : int
        Minimum number of points in a neighborhood for DBSCAN.
    eps : float
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other in DBSCAN.
    n_pcs : Optional[int]
        Number of principal components to use for PCA. If None, will be
        determined automatically using the elbow method.
    m : Optional[int]
        Number of cells to downsample from each batch. If None, all cells are
        used.
    n : Optional[int]
        Number of most variable genes to select. If None, all genes are used.
    embedding_method : str
        Method to use for embedding the data before DBSCAN. Options are 'PCA',
        'tSNE', or 'UMAP'. Default is 'PCA'.
    n_neighbors : int
        Number of neighbors to use for constructing the nearest neighbors graph.
    scale_max_val : int
        Maximum value for scaling the data. Default is 10.
    dendrogram_limit : int
        Limit for the dendrogram plot to show only the top levels.
    clustermap_limit : Optional[int]
        Limit for the clustermap to show only a subset of cells. If None, all
        cells are used.
    markers_per_clust : int
        Number of top marker genes to extract per cluster.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[str]
        List of top marker genes identified from the hierarchical clustering.
    """
    np.random.seed(seed)

    # --- Step 1: Preprocess and Batch correction ---
    print("--- Performing batch correction with Scanorama ---")
    new_batches = []
    for batata in batches:
        indices = np.random.choice(
            batata.shape[0], size=int(m/len(batches)), replace=False
        )
        batata = batata[indices, :]
        new_batches.append(batata)
        del batata
        gc.collect()
    batches = scanorama.correct_scanpy(
        new_batches, return_dimred=False, verbose=False
    )
    del new_batches
    gc.collect()
    adata = ad.concat(
        batches, axis=0, join="outer", label="batch", keys=batch_keys
    )
    del batches
    gc.collect()
    print("Batch correction completed.")
    if n:
        sc.pp.highly_variable_genes(adata, n_top_genes=n, flavor="seurat")
        adata = adata[:, adata.var.highly_variable].copy()
    adata.raw = adata.copy()
    sc.pp.scale(adata, max_value=scale_max_val, zero_center=False)
    if m:
        print(f"Downsampled to {m} cells")
    else:
        print(f"All cells used, {adata.n_obs}")
    if n:
        print(f"Selected top {n} most variable genes")
    else:
        print(f"All genes used, {adata.n_vars}")
    print("Preprocessing completed")
    print("---\n")

    # --- Step 2: PCA --- 
    print("--- Performing PCA ---")
    sc.tl.pca(adata, svd_solver="arpack", random_state=seed)
    sc.pl.pca_variance_ratio(adata, log=True)
    print("PCA completed")
    if not n_pcs:
        n_pcs = find_elbow_pcs(adata.uns["pca"]["variance_ratio"])
        print(
            "n_PCs for PCA not provided, "
            f"calculated using elbow method as: {n_pcs}"
        )
    print("---\n")

    # --- Step 3: Nearest neighbors distance matrix ---
    print("--- Finding nearest neighbors ---")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    print(
        "Calculated nearest neighbors distance matrix "
        f"using 'n_neighbors'={n_neighbors}"
    )
    print("---\n")

    # --- Step 4: Noise detection using DBSCAN ---
    print("--- Removing noisy cells using DBSCAN ---")
    dbscanner = DBSCAN(eps=eps, min_samples=min_pts)
    if embedding_method.lower() == "pca":
        embedding = adata.obsm["X_pca"][:, :n_pcs]
    elif embedding_method.lower() == "tsne":
        sc.tl.tsne(adata, random_state=seed)
        embedding = adata.obsm["X_tsne"]
    elif embedding_method.lower() == "umap":
        sc.tl.umap(adata, random_state=seed)
        embedding = adata.obsm["X_umap"]
    else:
        raise ValueError(
            f"Invalid embedding method for DBSCAN: {embedding_method} "
            f"select one from 'pca', 'tsne', or 'umap. "
        )
    print(f"Using {embedding_method} as embedding method")
    cluster_labels = dbscanner.fit_predict(embedding)
    print("DBSCAN ran, removing noise cells...")
    noise_mask = cluster_labels == -1
    adata = adata[~noise_mask, :].copy()
    print(f"Noisy cells removed: {noise_mask.sum()}")
    print("---\n")

    # --- Step 5: Hierarchical clustering ---
    print("--- Clustering cells ---")
    # Step 5.1: Pairwise distances
    X = adata.obsm["X_pca"][:, :30]
    Y = pdist(X)

    # Step 5.2: Compute linkage matrix
    Z = sch.linkage(Y, method="ward")  # I think ward is always the best
                                       # and should work nicely here as it
                                       # minimizes variance by each clustering
                                       # step

    # Step 5.3: Get flat clusters
    print(f"--- Extracting {num_clusters} clusters from dendrogram ---")
    # Cut the dendrogram tree to get flat clusters
    cluster_assignments = sch.fcluster(Z, t=num_clusters, criterion="maxclust")
    cluster_assignments_str = [f"Cluster_{i}" for i in cluster_assignments]
    adata.obs["hierarchical_clusters"] = pd.Categorical(cluster_assignments_str)
    c, _ = cophenet(Z, Y)
    print(f"Cophenetic Correlation Coefficient: {c:.4f}\n")

    # Step 5.4: Marker genes
    sc.tl.rank_genes_groups(
        adata,
        "hierarchical_clusters",
        method="wilcoxon",
        key_added="rank_genes_hclust"
    )
    # Debug: Check if rank_genes_groups worked
    print(
        "Rank genes groups results keys:", adata.uns["rank_genes_hclust"].keys()
    )
    # Get marker genes more robustly
    try:
        marker_df = pd.DataFrame(adata.uns["rank_genes_hclust"]["names"])
        print(f"Marker genes dataframe shape: {marker_df.shape}")
        print("First few rows of marker genes:")
        print(marker_df.head())
        # Get top markers, ensuring we have valid genes
        top_markers = []
        for col in marker_df.columns:
            # Top 5 from each cluster
            top_markers.extend(marker_df[col].head(markers_per_clust).tolist())
        # Remove duplicates and None values
        top_markers = [
            gene for gene in list(set(top_markers))
            if gene is not None and str(gene) != "nan"
        ]
        print(f"Total unique top marker genes found: {len(top_markers)}")
        if len(top_markers) == 0:
            print(
                "WARNING: No marker genes found! "
                "Using most variable genes instead..."
            )
            # Fallback to highly variable genes
            if hasattr(adata.var, "highly_variable"):
                top_markers = list(
                    adata.var_names[adata.var.highly_variable][:20]
                )
            else:
                # Ultimate fallback - just use first 20 genes
                top_markers = list(adata.var_names[:20])
            print(f"Using {len(top_markers)} fallback genes")
    except Exception as e:
        print(f"Error extracting marker genes: {e}")
        print("Using first 20 genes as fallback...")
        top_markers = list(adata.var_names[:20])
    print(f"Final marker genes list length: {len(top_markers)}")
    print(f"First 10 marker genes: {top_markers[:10]}")
    print("---\n")

    # --- Step 6: Plotting ---
    print("--- Plotting ---")
    # Step 6.1 Plot truncated dendrogram
    color_threshold = Z[-num_clusters+1, 2]
    plt.figure(figsize=(12, 6))
    sch.dendrogram(
        Z,
        truncate_mode="lastp",
        p=dendrogram_limit,
        leaf_rotation=90,
        orientation="top",
        color_threshold=color_threshold,
        above_threshold_color="grey",
    )
    plt.title(f"Truncated Dendrogram (Top {dendrogram_limit} Levels)")
    plt.xlabel("Clustered Cells")
    plt.ylabel("Distance")
    plt.show()

    # Step 6.2: Plot a clustermap
    if clustermap_limit:
        idx = random.sample(range(X.shape[0]), clustermap_limit)
        X_clustermap = X[idx, :]
    sns.clustermap(
        X_clustermap if clustermap_limit else X,
        method="ward", metric="euclidean",
        cmap="bwr", figsize=(10, 10)
    )
    plt.title("Clustermap of Sampled Cells")
    plt.show()

    # Step 6.3: better plots with marker genes
    print(f"Top marker genes for visualization: {top_markers[:10]}...")
    plot_dendro_heatmap_with_clusters(
        adata=adata,
        Z=Z,
        cluster_labels_key="hierarchical_clusters",
        dendrogram_title=f"Hierarchical Clustering (m={m}, n={n})",
        marker_genes=top_markers,
        num_clusters=num_clusters
    )
    sc.pl.rank_genes_groups_heatmap(
        adata,
        n_genes=markers_per_clust,
        key="rank_genes_hclust",
        groupby="hierarchical_clusters",
        show_gene_labels=True,
        # use_raw=True,
        cmap="bwr",
        show=False,
        dendrogram=True
    ) 
    plt.show()
    # test
    plot_combined_clustermap(
        adata=adata,
        Z_cells=Z,  # Pass the cell linkage matrix
        cluster_labels_key="hierarchical_clusters",
        batch_key="batch",  # This key is set during adata concatenation
        plot_title=f"Hierarchical Clustermap (m={m}, n={n})",
        marker_genes=top_markers,
        num_clusters_for_coloring=num_clusters
    )
    # end test
    # Step 6.4: Plot UMAP with clusters and marker genes
    try:
        sc.tl.umap(adata, random_state=seed)
    except UserWarning as e:
        print(e)
        print("Increasing number of neighbors to 30 for UMAP...")
        sc.pp.neighbors(adata, n_neighbors=30, n_pcs=n_pcs)
    colors = ["hierarchical_clusters"] + top_markers
    titles = ["Hierarchical Clusters"] + top_markers
    sc.pl.umap(
        adata,
        color=colors,
        title=titles,
        show=False
    )
    plt.suptitle("UMAP with Hierarchical Clusters and Top Marker Genes")
    plt.show()
    print("---\n")

    # --- Step 7: Cleanup ---
    del adata
    if clustermap_limit:
        del X_clustermap
        del idx
    del X
    del Y
    del Z
    gc.collect()

    return top_markers