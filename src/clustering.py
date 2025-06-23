# import scanpy as sc
# import numpy as np
# import pandas as pd

# from sklearn.cluster import DBSCAN
# from sklearn.metrics import calinski_harabasz_score

# import optuna
# from optuna.distributions import IntDistribution, FloatDistribution

# import tqdm

# from typing import Optional


# class ClusteringOptimization:

#     def __init__(
#             self,
#             adata: sc.AnnData,
#             genes_to_clust: tuple[str, Optional[int]] = ("top", 2_000),
#             max_value_scale: int = 10,
#             tune_dbscan: bool = True,
#             tune_leiden: bool = True,
#             tune_neighbors: bool = True,
#             dbscan_params: Optional[dict[str, int]] = None,
#             leiden_params: Optional[dict[str, float]] = None,
#             neighbors_params: Optional[dict[str, int]] = None
#         ):
#         self.adata = adata
#         self.genes_to_clust = genes_to_clust[0]
#         self.n_genes_to_clust = genes_to_clust[1]
#         self.max_value_scale = max_value_scale
#         self.tune_dbscan = tune_dbscan
#         self.tune_leiden = tune_leiden
#         self.tune_neighbors = tune_neighbors
#         if not dbscan_params:
#             dbscan_params = {
#                 "eps": (1, 6),
#                 "min_samples": (1, 100)
#             }
#         else:
#             self.dbscan_params = dbscan_params
#         if not leiden_params:
#             leiden_params = {
#                 "resolution": (0.1, 1.0)
#             }
#         else:
#             self.leiden_params = leiden_params
#         if not neighbors_params:
#             neighbors_params = {
#                 "n_neighbors": (5, 50)
#             }
#         else:
#             self.neighbors_params = neighbors_params

#     def _define_hyperparameters(self):
#         r = {}
#         if self.tune_dbscan:
#             r["dbscan__eps"] = FloatDistribution(
#                 low=self.dbscan_params["eps"][0],
#                 high=self.dbscan_params["eps"][1],
#                 log=False,
#             )
#             r["dbscan__min_samples"] = IntDistribution(
#                 low=self.dbscan_params["min_samples"][0],
#                 high=self.dbscan_params["min_samples"][1],
#                 log=False,
#             )
#         if self.tune_leiden:
#             r["leiden__resolution"] = FloatDistribution(
#                 low=self.leiden_params["resolution"][0],
#                 high=self.leiden_params["resolution"][1],
#                 log=False,
#             )
#         if self.tune_neighbors:
#             ["neighbors__n_neighbors"] = IntDistribution(
#                 low=self.neighbors_params["n_neighbors"][0],
#                 high=self.neighbors_params["n_neighbors"][1],
#                 log=True,
#             )
#         if len(r) == 0:
#             raise ValueError("No hyperparameters defined for optimization.")
#         return r

#     def _objective(self):
#         # optimize based on: chi = calinski_harabasz_score(X, labels)
#         pass

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score

import optuna

from typing import Optional, Dict, Tuple, Any

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ClusteringOptimization:
    """
    A class to streamline and optimize scRNA-seq clustering using Optuna.

    This class encapsulates the preprocessing, clustering (DBSCAN noise removal + Leiden),
    and hyperparameter tuning steps for single-cell analysis. It uses the
    Calinski-Harabasz score as the optimization metric to find the best set of
    parameters for `n_neighbors`, DBSCAN's `eps` and `min_samples`, and
    Leiden's `resolution`.

    Args:
        adata (sc.AnnData): The annotated data matrix.
        genes_to_clust (Tuple[str, Optional[int]]): Method and number of genes for
            clustering. E.g., ("top", 2000) for top 2000 highly variable genes.
        max_value_scale (int): The value to clip at during `sc.pp.scale`.
        n_pcs (Optional[int]): Number of principal components to use. If None, it
            will be determined automatically by finding the 'elbow'.
        tune_dbscan (bool): If True, tune DBSCAN parameters (`eps`, `min_samples`).
        tune_leiden (bool): If True, tune Leiden parameter (`resolution`).
        tune_neighbors (bool): If True, tune the number of neighbors (`n_neighbors`).
        dbscan_params (Optional[Dict[str, Tuple[float, float]]]): Search range for DBSCAN
            parameters. Keys: 'eps', 'min_samples'.
        leiden_params (Optional[Dict[str, Tuple[float, float]]]): Search range for
            Leiden's resolution. Key: 'resolution'.
        neighbors_params (Optional[Dict[str, Tuple[int, int]]]): Search range for
            the number of neighbors. Key: 'n_neighbors'.
    """

    def __init__(
        self,
        adata: sc.AnnData,
        genes_to_clust: Tuple[str, Optional[int]] = ("top", 2000),
        max_value_scale: int = 10,
        n_pcs: Optional[int] = None,
        tune_dbscan: bool = True,
        tune_leiden: bool = True,
        tune_neighbors: bool = True,
        dbscan_params: Optional[Dict[str, Tuple[float, float]]] = None,
        leiden_params: Optional[Dict[str, Tuple[float, float]]] = None,
        neighbors_params: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        self.initial_adata = adata.copy()
        self.genes_to_clust = genes_to_clust[0]
        self.n_genes_to_clust = genes_to_clust[1]
        self.max_value_scale = max_value_scale
        self.n_pcs = n_pcs

        self.tune_dbscan = tune_dbscan
        self.tune_leiden = tune_leiden
        self.tune_neighbors = tune_neighbors

        # --- Set hyperparameter search spaces with defaults ---
        self.dbscan_params = dbscan_params if dbscan_params is not None else {
            "eps": (0.5, 5.0),
            "min_samples": (5, 50)
        }
        self.leiden_params = leiden_params if leiden_params is not None else {
            "resolution": (0.1, 1.0)
        }
        self.neighbors_params = neighbors_params if neighbors_params is not None else {
            "n_neighbors": (5, 50)
        }

        self.study: Optional[optuna.Study] = None
        self.processed_adata: Optional[sc.AnnData] = None

    def _find_elbow_pcs(self, variance_ratio: np.ndarray, n_points_to_fit: int = 5) -> int:
        """Heuristically finds the 'elbow' in the PCA variance ratio plot."""
        # It finds the point with the maximum distance to a line drawn
        # from the first to the last point of the curve.
        y = variance_ratio
        x = np.arange(len(y))

        # Line from first to last point
        line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

        # Vector from first point to all other points
        vec_from_first = np.array([x - x[0], y - y[0]]).T

        # Project vectors onto the line's normal
        # The distance is the length of the projection onto the normal vector
        normal_vec = np.array([-line_vec_norm[1], line_vec_norm[0]])
        dist_from_line = np.abs(np.dot(vec_from_first, normal_vec))

        # The elbow is the point with the maximum distance
        elbow_index = np.argmax(dist_from_line)
        # Return as 1-based count
        return elbow_index + 1


    def _preprocess_data(self):
        """
        Runs the initial preprocessing steps: HVG selection, scaling, and PCA.
        This is done once before the optimization starts.
        """
        print("--- Starting Preprocessing ---")
        adata = self.initial_adata.copy()

        print(f"Finding top {self.n_genes_to_clust} highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=self.n_genes_to_clust, flavor="seurat")
        adata = adata[:, adata.var.highly_variable].copy()

        print("Scaling data...")
        sc.pp.scale(adata, max_value=self.max_value_scale)

        print("Performing PCA...")
        sc.tl.pca(adata, svd_solver='arpack')

        if self.n_pcs is None:
            print("Automatically determining number of PCs...")
            self.n_pcs = self._find_elbow_pcs(adata.uns['pca']['variance_ratio'])
            print(f"Found elbow at {self.n_pcs} PCs.")

        self.processed_adata = adata
        print("--- Preprocessing Complete ---")

    def _define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Defines the hyperparameter search space for Optuna."""
        params = {}
        if self.tune_dbscan:
            params["eps"] = trial.suggest_float(
                "dbscan__eps", self.dbscan_params["eps"][0], self.dbscan_params["eps"][1]
            )
            params["min_samples"] = trial.suggest_int(
                "dbscan__min_samples", self.dbscan_params["min_samples"][0], self.dbscan_params["min_samples"][1]
            )
        if self.tune_leiden:
            params["resolution"] = trial.suggest_float(
                "leiden__resolution", self.leiden_params["resolution"][0], self.leiden_params["resolution"][1]
            )
        if self.tune_neighbors:
            params["n_neighbors"] = trial.suggest_int(
                "neighbors__n_neighbors", self.neighbors_params["n_neighbors"][0], self.neighbors_params["n_neighbors"][1], log=True
            )

        if not params:
            raise ValueError("No hyperparameters selected for tuning. Set at least one 'tune_*' flag to True.")

        return params

    def _objective(self, trial: optuna.Trial) -> float:
        """
        The objective function for Optuna optimization.

        For a given trial, it runs the clustering pipeline and returns the
        Calinski-Harabasz score.
        """
        params = self._define_hyperparameter_space(trial)
        adata_trial = self.processed_adata.copy()

        # Use PCA embedding up to the determined number of PCs
        embedding = adata_trial.obsm["X_pca"][:, :self.n_pcs]

        # --- 1. DBSCAN Noise Filtering ---
        db = DBSCAN(
            eps=params.get("eps", 2.0), # Default if not tuning
            min_samples=params.get("min_samples", 15) # Default if not tuning
        ).fit(embedding)
        labels = db.labels_

        # Create a mask for non-noise points
        non_noise_mask = labels != -1

        # If all points are noise or less than 2 cells remain, it's a bad run
        if np.sum(non_noise_mask) < 2:
            return 0.0

        adata_filtered = adata_trial[non_noise_mask].copy()

        # --- 2. Leiden Clustering on Filtered Data ---
        # Recompute neighbors on the filtered data
        sc.pp.neighbors(
            adata_filtered,
            n_neighbors=params.get("n_neighbors", 15), # Default if not tuning
            n_pcs=self.n_pcs
        )

        # Run Leiden
        sc.tl.leiden(
            adata_filtered,
            resolution=params.get("resolution", 0.4), # Default if not tuning
            flavor="igraph",
            n_iterations=2,
            directed=False
        )

        leiden_labels = adata_filtered.obs["leiden"]

        # Calinski-Harabasz score is undefined for 1 cluster.
        if len(set(leiden_labels)) < 2:
            return 0.0

        # --- 3. Evaluate Clustering ---
        # The score should be calculated on the same data that was clustered
        filtered_embedding = adata_filtered.obsm["X_pca"][:, :self.n_pcs]
        score = calinski_harabasz_score(filtered_embedding, leiden_labels)

        return score

    def run_optimization(self, n_trials: int = 100, storage: Optional[str] = None, study_name: Optional[str] = None):
        """
        Runs the full Optuna optimization study.

        Args:
            n_trials (int): The number of optimization trials to run.
            storage (Optional[str]): Database URL for study storage (e.g., 'sqlite:///db.sqlite3').
            study_name (Optional[str]): Name for the study, required for distributed optimization.
        """
        if self.processed_adata is None:
            self._preprocess_data()

        self.study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )

        print(f"\n--- Running Optuna Optimization for {n_trials} trials ---")
        self.study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)

        print("\n--- Optimization Complete ---")
        print(f"Best trial number: {self.study.best_trial.number}")
        print(f"Best Score (Calinski-Harabasz): {self.study.best_value:.4f}")
        print("Best Parameters:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    def get_best_params(self) -> Dict[str, Any]:
        """Returns the best hyperparameters found by the study."""
        if self.study is None:
            raise RuntimeError("Optimization has not been run yet. Call `run_optimization()` first.")
        return self.study.best_params

    def apply_best_params(self) -> sc.AnnData:
        """
        Applies the best found parameters to the data and returns the final AnnData object.

        This method re-runs the pipeline with the optimal parameters and saves the
        resulting clustering and UMAP coordinates to the original `adata` object.
        """
        if self.study is None:
            raise RuntimeError("Optimization has not been run yet. Call `run_optimization()` first.")

        print("\n--- Applying Optimal Parameters to Data ---")
        best_params = self.get_best_params()

        # Start with a fresh copy of the initial data
        final_adata = self.initial_adata.copy()

        # Re-run preprocessing
        print("1. Re-running preprocessing...")
        sc.pp.highly_variable_genes(final_adata, n_top_genes=self.n_genes_to_clust, flavor="seurat")
        final_adata = final_adata[:, final_adata.var.highly_variable].copy()
        sc.pp.scale(final_adata, max_value=self.max_value_scale)
        sc.tl.pca(final_adata, svd_solver="arpack")

        embedding = final_adata.obsm["X_pca"][:, :self.n_pcs]

        # 2. Run DBSCAN with optimal parameters to find noise
        print("2. Filtering noise with optimal DBSCAN parameters...")
        db_optimal = DBSCAN(
            eps=best_params.get("dbscan__eps", 2.0),
            min_samples=best_params.get("dbscan__min_samples", 15)
        ).fit(embedding)

        non_noise_mask = db_optimal.labels_ != -1
        final_adata.obs["dbscan_labels"] = [f"Cluster {l}" if l != -1 else "Noise" for l in db_optimal.labels_]

        print(f"   Removed {(~non_noise_mask).sum()} noise points.")
        print(f"   Retained {non_noise_mask.sum()} cells for final clustering.")

        # Create filtered dataset
        adata_filtered = final_adata[non_noise_mask].copy()

        # 3. Re-compute neighbors and run Leiden on filtered data
        print("3. Running Leiden clustering on noise-filtered data...")
        sc.pp.neighbors(
            adata_filtered,
            n_neighbors=best_params.get("neighbors__n_neighbors", 15),
            n_pcs=self.n_pcs
        )
        sc.tl.leiden(
            adata_filtered,
            resolution=best_params.get("leiden__resolution", 0.4),
            flavor="igraph"
        )

        # 4. Transfer results back to the main AnnData object
        print("4. Finalizing results...")
        final_adata.obs["optimized_leiden"] = "Filtered_Out_Noise"
        final_adata.obs.loc[non_noise_mask, "optimized_leiden"] = adata_filtered.obs["leiden"].values

        # 5. Compute UMAP for visualization
        print("5. Computing UMAP...")
        # First on the full data to see noise
        sc.pp.neighbors(final_adata, n_pcs=self.n_pcs)
        sc.tl.umap(final_adata)
        # Then on the filtered data for a cleaner view
        sc.tl.umap(adata_filtered)

        # Store the filtered UMAP coordinates in the main object for plotting
        final_adata.obsm["X_umap_filtered"] = np.full((final_adata.n_obs, 2), np.nan)
        final_adata.obsm["X_umap_filtered"][non_noise_mask] = adata_filtered.obsm["X_umap"]

        print("\n--- Final AnnData object is ready ---")
        print("Results are stored in:")
        print("  - `adata.obs['dbscan_labels']`: Noise classification from DBSCAN.")
        print("  - `adata.obs['optimized_leiden']`: Final Leiden clusters after noise removal.")
        print("  - `adata.obsm['X_umap']`: UMAP on all cells.")
        print("  - `adata.obsm['X_umap_filtered']`: UMAP on non-noise cells only.")

        return final_adata