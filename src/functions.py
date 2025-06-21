import scanpy as sc
import pandas as pd
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from pathlib import Path
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

warnings.filterwarnings("ignore")


def download_data(brain_parts,directories_to_use,metadata_to_use):
    download_base = Path('../data/abc_atlas')
    abc_cache = AbcProjectCache.from_cache_dir(download_base)
    
    print('Current manifest is ',abc_cache.current_manifest)
    print('All available manifests: ',abc_cache.list_manifest_file_names)

    abc_cache.load_manifest('releases/20230630/manifest.json')
    print("We will be using manifest :", abc_cache.current_manifest)
    print("All available data directories:",abc_cache.list_directories)
    print("Directories to be used:", directories_to_use)
    # print("We will be using WMB-10Xv3 and WMB-10Xv2 directories \n")

    for directory in directories_to_use:
        print(f"All available data files of {directory} directory: {abc_cache.list_data_files(directory)}")

    
    # print("All available data files of WMB-10Xv3 directory: ",abc_cache.list_data_files('WMB-10Xv3'))
    # print("All available data files of WMB-10Xv2 directory: ",abc_cache.list_data_files('WMB-10Xv2'))
    print("Metadata to be used:",abc_cache.list_metadata_files(metadata_to_use))
    
    print("Downloading the metadata\n")
    # abc_cache.get_directory_metadata('WMB-10X')
    abc_cache.get_directory_metadata(metadata_to_use)
    
    for directory in directories_to_use:
        for brain_part in brain_parts:
            print(f"Dowloading the {directory} data file for brain part : {brain_part}")
            # fnamev3='WMB-10Xv3-'+ brain_part + '/log2'
            # abc_cache.get_data_path(directory='WMB-10Xv3', file_name=fnamev3)
            
            # fnamev2='WMB-10Xv2-'+ brain_part + '/log2'
            # abc_cache.get_data_path(directory='WMB-10Xv2', file_name=fnamev2)
            fname= directory + '-' + brain_part + '/log2'
            abc_cache.get_data_path(directory=directory, file_name=fname)

            # print(f"Dowloading the WMB-10Xv2 data file for brain part : {brain_part}")