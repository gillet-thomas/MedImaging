import warnings
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from nilearn import datasets, plotting, connectome, image
from nilearn.regions import RegionExtractor
from nilearn.decomposition import DictLearning

def load_data(external=False):
    if external:
        rest_dataset = datasets.fetch_development_fmri(n_subjects=20)
        func_filenames = rest_dataset.func
        confounds = rest_dataset.confounds
    else:
        func_filenames = sorted(glob('data/resampled/*.nii'))
        confounds = [None] * len(func_filenames)
        print(f"Found {len(func_filenames)} fMRI files")
    
    return func_filenames, confounds

def create_brain_maps(func_filenames):
    dict_learn = DictLearning(
        n_components=8,
        smoothing_fwhm=6.0,
        memory="nilearn_cache",
        memory_level=2,
        random_state=0,
        standardize="zscore_sample",
    )
    
    dict_learn.fit(func_filenames)
    components_img = dict_learn.components_img_
    
    return dict_learn, components_img

def voxel_wise_connectivity(func_filenames, confounds, mask_strategy='epi'):
    # Initialize NiftiMasker for voxel-wise analysis
    masker = NiftiMasker(
        standardize='zscore_sample',
        memory="nilearn_cache", 
        memory_level=1,
        mask_strategy=mask_strategy,
        smoothing_fwhm=6
    )
    
    # Fit the masker to all functional images
    masker.fit(func_filenames)
    print(f"Mask shape: {masker.mask_img_.shape}")
    correlations = []
    connectome_measure = connectome.ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample"
    )
    
    for filename, confound in zip(func_filenames, confounds):
        # Extract time series for each voxel
        voxel_time_series = masker.transform(filename, confounds=confound)
        
        # Calculate correlation matrix
        # Note: This will be a large matrix (n_voxels x n_voxels)
        correlation = connectome_measure.fit_transform([voxel_time_series])
        correlations.append(correlation)
    
    # Average correlation matrices across subjects
    mean_correlations = np.mean(correlations, axis=0).squeeze()
    
    # Get the mask image for visualization
    mask_img = masker.mask_img_
    
    return mean_correlations, mask_img, masker

def visualize_voxel_connectivity(mean_correlations, mask_img, masker, threshold=0.5):
    # Create a connectivity map for a seed voxel
    # Here we'll use the middle voxel as an example
    middle_voxel_idx = mean_correlations.shape[0] // 2
    seed_connectivity = mean_correlations[middle_voxel_idx]
    
    # Transform correlation vector back to brain space
    seed_connectivity_map = masker.inverse_transform(seed_connectivity.reshape(1, -1))
    
    # Plot the connectivity map
    display = plotting.plot_stat_map(
        seed_connectivity_map,
        threshold=threshold,
        title="Voxel-wise connectivity map (from seed voxel)",
        cut_coords=(0, 0, 0)
    )
    return display

if __name__ == "__main__":
    # Load data
    func_filenames, confounds = load_data(False)
    
    # Optional: Create brain maps for comparison
    dict_learn, components_img = create_brain_maps(func_filenames)
    
    # Perform voxel-wise connectivity analysis
    mean_correlations, mask_img, masker = voxel_wise_connectivity(func_filenames, confounds)
    
    print(f"Correlation matrix shape: {mean_correlations.shape}")
    
    # Visualize connectivity for a seed voxel
    display = visualize_voxel_connectivity(mean_correlations, mask_img, masker)
    plotting.show()
    
    # Optional: Create a sparse connectivity matrix by thresholding
    threshold = 0.5
    sparse_correlations = np.where(np.abs(mean_correlations) > threshold, mean_correlations, 0)
    
    # Plot a subset of the correlation matrix (it's too large to plot entirely)
    subset_size = 1000
    title = f"Correlation matrix (showing {subset_size}x{subset_size} voxels)"
    plotting.plot_matrix(
        mean_correlations[:subset_size, :subset_size],
        vmax=1,
        vmin=-1,
        colorbar=True,
        title=title
    )
    plotting.show()