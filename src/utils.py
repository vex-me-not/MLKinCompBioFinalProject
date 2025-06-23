import scanpy as sc
import numpy as np
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
    if data_path.endswith('.h5ad'):
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