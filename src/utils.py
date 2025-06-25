import scanpy as sc
import numpy as np
from mygene import MyGeneInfo
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
import warnings


def download_data(
        download_base,
        brain_parts,
        directories_to_use,
        metadata_to_use,
        log2=True,
        verbose=True
    ):
    """
    Download data from the ABC Atlas project cache.

    Parameters
    ----------
    download_base : str
        Base directory for downloading data.
    brain_parts : list of str
        List of brain parts to download data for.
    directories_to_use : list of str
        List of directories to use for downloading data.
    metadata_to_use : list of str
        List of metadata files to download.
    log2 : bool, optional
        If True, download log2 transformed data; otherwise, download raw data.
        Default is True.
    """
    warnings.filterwarnings("ignore")
    abc_cache = AbcProjectCache.from_cache_dir(download_base)

    if verbose:
        print("Current manifest is ", abc_cache.current_manifest)
        print("All available manifests: ", abc_cache.list_manifest_file_names)

    abc_cache.load_manifest("releases/20230630/manifest.json")

    if verbose:
        print("We will be using manifest :", abc_cache.current_manifest)
        print("All available data directories: ", abc_cache.list_directories)
        print("Directories to be used: ", directories_to_use)

    for directory in directories_to_use:
        if verbose:
            print(
                f"All available data files of {directory} directory: "
                f"{abc_cache.list_data_files(directory)}"
            )

    if verbose:
        print(
            "Metadata to be used: ",
            f"{abc_cache.list_metadata_files(metadata_to_use)}"
        )
        print("Downloading the metadata\n")

    abc_cache.get_directory_metadata(metadata_to_use)

    for directory in directories_to_use:
        for brain_part in brain_parts:
            if verbose:
                print(
                    f"Dowloading the {directory} "
                    f"data file for brain part: {brain_part}"
                )

            if log2:
                fname = directory + "-" + brain_part + "/log2"
            else:
                fname = directory + "-" + brain_part + "/raw"
            abc_cache.get_data_path(directory=directory, file_name=fname)


def load_data(data_path, backed=None):
    """
    Load data from the specified path and format.

    Parameters
    ----------
    data_path : str
        Path to the data file (should be .h5ad format).
    backed : bool, optional
        If True, load the data in backed mode (default is None, which means
        not backed).

    Returns
    -------
    adata : AnnData
        The loaded AnnData object.
    """
    if data_path.endswith(".h5ad"):
        adata = sc.read_h5ad(data_path, backed=backed)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .h5ad file."
        )
    return adata


def find_elbow_pcs(variance_ratio):
    """
    Heuristically finds the 'elbow' in the PCA variance ratio plot.

    The elbow is the point furthest from the line connecting the first and last
    points in the variance ratio array. This point indicates the optimal number
    of principal components to retain, balancing explained variance and model
    complexity.

    Parameters
    ----------
    variance_ratio : np.ndarray
        An array of variance ratios from PCA, typically the explained variance
        for each principal component.

    Returns
    -------
    int
        The index of the elbow point in the variance ratio array, indicating
        the optimal number of principal components to retain.
        The index is returned as a 1-based index.
    """
    y = variance_ratio
    x = np.arange(len(y))

    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    vec_from_first = np.array([x - x[0], y - y[0]]).T

    normal_vec = np.array([-line_vec_norm[1], line_vec_norm[0]])
    dist_from_line = np.abs(np.dot(vec_from_first, normal_vec))

    elbow_index = np.argmax(dist_from_line)

    return elbow_index + 1


def annotate_genes_with_go_and_pathways(ensembl_ids, top_n=10):
    mg = MyGeneInfo()
    results = []

    def extract_go_and_pathways(gene_data):
        go_terms = []
        pathways = []

        go = gene_data.get("go", {})
        for cat in ["BP", "MF", "CC"]:
            for entry in go.get(cat, []):
                go_terms.append(
                    f"{cat}: {entry.get("term")} ({entry.get("id")})"
                )

        pwy = gene_data.get("pathway", {})
        if isinstance(pwy, dict):
            for db, content in pwy.items():
                if isinstance(content, list):
                    for entry in content:
                        pathways.append(
                            f"{db.upper()}: {entry.get("name")} ({entry.get("id")})"
                        )
                elif isinstance(content, dict):
                    pathways.append(
                        f"{db.upper()}: {content.get("name")} ({content.get("id")})"
                    )

        return "; ".join(go_terms), "; ".join(pathways)

    # Query one-by-one
    for gene in ensembl_ids:
        try:
            data = mg.getgene(
                gene, fields="symbol,name,go,pathway", species="mouse"
            )
            go_str, pwy_str = extract_go_and_pathways(data)
            results.append({
                "Ensembl_ID": gene,
                "Symbol": data.get("symbol", ""),
                "Gene_Name": data.get("name", ""),
                "GO_Terms": go_str,
                "Pathways": pwy_str
            })
        except Exception as e:
            print(f"Error retrieving {gene}: {e}")
            results.append({
                "Ensembl_ID": gene,
                "Symbol": "",
                "Gene_Name": "",
                "GO_Terms": "",
                "Pathways": ""
            })

    annotations_df = pd.DataFrame(results)

    all_go_terms = []
    for terms in annotations_df["GO_Terms"]:
        if pd.notna(terms):
            all_go_terms.extend(term.strip() for term in terms.split(";"))

    go_counts = Counter(all_go_terms)

    #GO terms shared by 2 or more genes
    shared_go = {term: count for term, count in go_counts.items() if count > 1}

    all_pathways = []
    for p in annotations_df["Pathways"]:
        if pd.notna(p):
            all_pathways.extend(p.strip() for p in p.split(";"))

    pathway_counts = Counter(all_pathways)
    shared_pathways = {
        p: count for p, count in pathway_counts.items() if count > 1
    }


    def plot_shared_terms(term_dict, title, top_n):
        # Filter out empty or None keys before sorting
        clean_terms = {
            k: v for k, v in term_dict.items() if k not in (None, "") and v is not None
        }

        # Sort and take top_n
        top_terms = sorted(clean_terms.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Safely unpack if not empty
        if top_terms:
            ks, vs = zip(*top_terms)
        else:
            ks, vs = [], []

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(ks, vs)
        plt.xlabel("Number of Genes")
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    plot_shared_terms(shared_go, "Top Shared GO Terms", top_n=top_n)
    plot_shared_terms(shared_pathways, "Top Shared Pathways", top_n=top_n)


    return annotations_df