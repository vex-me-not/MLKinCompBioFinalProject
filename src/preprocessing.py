import scanpy as sc
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns


def describe_adata(adata):
    """
    Print basic information about the AnnData object.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to describe.
    """
    print(f"Number of observations: {adata.n_obs}")
    print(f"Number of variables: {adata.n_vars}")
    print(f"Shape of the data: {adata.X.shape}")
    print("Is backed") if adata.isbacked else print("Not backed")


def tsne_compare(
        adata1,
        adata2,
        name_1,
        name_2,
        title=None,
        n_cells_per_dataset=5_000,
        n_comps=50,
        n_pcs=40,
        perplexity=30,
        palette=["#0095ff", "#fd7600"],
        size=30,
        alpha=0.8,
        legend_loc="right margin",
        seed=42
    ):
    """
    Perform t-SNE on two datasets and plot the results.

    Parameters
    ----------
    adata1 : AnnData
        The first AnnData object.
    adata2 : AnnData
        The second AnnData object.
    name_1 : str
        Name for the first dataset.
    name_2 : str
        Name for the second dataset.
    title : str, default=None
        Title for the plot. If None, a default title will be used.
    n_cells_per_dataset : int, default=5000
        Number of cells to sample from each dataset.
    n_comps : int, default=50
        Number of principal components to compute.
    n_pcs : int, default=40
        Number of principal components to use for t-SNE.
    perplexity : int, default=30
        Perplexity parameter for t-SNE.
    palette : list, default=["#0095ff", "#fd7600"]
        Color palette for the datasets in the plot.
    size : int, default=30
        Size of the points in the t-SNE plot.
    alpha : float, default=0.8
        Transparency of the points in the t-SNE plot.
    legend_loc : str, default="right margin"
        Location of the legend in the plot.
    seed : int, default=42
        Random seed for reproducibility.
    """
    np.random.seed(seed)
    a1_indices = np.random.choice(
        adata1.n_obs, size=min(n_cells_per_dataset, adata1.n_obs), replace=False
    )
    a2_indices = np.random.choice(
        adata2.n_obs, size=min(n_cells_per_dataset, adata1.n_obs), replace=False
    )

    a1_subset = adata1[a1_indices, :].to_memory()
    a2_subset = adata2[a2_indices, :].to_memory()

    a1_subset.obs["dataset"] = name_1
    a2_subset.obs["dataset"] = name_2

    adata_combined = ad.concat(
        [a1_subset, a2_subset], 
        label="batch", 
        keys=[name_1, name_2],
        index_unique="-"
    )

    sc.tl.pca(adata_combined, n_comps=n_comps, svd_solver="arpack")

    sc.tl.tsne(
        adata_combined, n_pcs=n_pcs, perplexity=perplexity, random_state=seed
    )

    if not title:
        title = f"t-SNE of {name_1} and {name_2}"

    sc.pl.tsne(
        adata_combined,
        color="dataset",
        palette=palette,
        size=size,
        alpha=alpha,
        legend_loc=legend_loc,
        title=title
    )

    plt.show()

    del a1_indices
    del a2_indices
    del a1_subset
    del a2_subset
    del adata_combined


def qc_inspect(
        adata_mem,
        n_top_genes=2000,
    ):
    # qc metrics
    sc.pp.calculate_qc_metrics(adata_mem, inplace=True)

    # violin plots
    sc.pl.violin(
        adata_mem,
        keys=[
            "n_genes_by_counts",
            "total_counts",
            "pct_counts_in_top_100_genes"
        ],
        groupby="anatomical_division_label",
        jitter=0.4,
        stripplot=False
    )

    # boxplot
    sns.boxplot(
        data=adata_mem.obs,
        x="anatomical_division_label",
        y="n_genes_by_counts"
    )

    # highly variable genes
    sc.pp.highly_variable_genes(
        adata_mem, flavor="seurat", n_top_genes=n_top_genes
    )
    sc.pl.highly_variable_genes(adata_mem)

    # scatter plot of total counts vs pct counts in top 100 genes
    sc.pl.scatter(
        adata_mem,
        x="total_counts",
        y="pct_counts_in_top_100_genes",
    )

    # delete the adata_mem object to free memory
    del adata_mem