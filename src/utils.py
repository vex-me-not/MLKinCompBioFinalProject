import scanpy as sc
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache


def download_data(
        download_base,
        brain_parts,
        directories_to_use,
        metadata_to_use,
        log2=True
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
    abc_cache = AbcProjectCache.from_cache_dir(download_base)
    
    print("Current manifest is ", abc_cache.current_manifest)
    print("All available manifests: ", abc_cache.list_manifest_file_names)

    abc_cache.load_manifest("releases/20230630/manifest.json")
    print("We will be using manifest :", abc_cache.current_manifest)
    print("All available data directories: ", abc_cache.list_directories)
    print("Directories to be used: ", directories_to_use)

    for directory in directories_to_use:
        print(
            f"All available data files of {directory} directory: "
            f"{abc_cache.list_data_files(directory)}"
        )

    print(
        "Metadata to be used:", abc_cache.list_metadata_files(metadata_to_use)
    )

    print("Downloading the metadata\n")
    abc_cache.get_directory_metadata(metadata_to_use)

    for directory in directories_to_use:
        for brain_part in brain_parts:
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