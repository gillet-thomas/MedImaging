import numpy as np
from nilearn import datasets, input_data, plotting, connectome, image
from nilearn.maskers import NiftiMapsMasker
from nilearn.regions import RegionExtractor
from nilearn import plotting
from glob import glob
import os
from nilearn.decomposition import DictLearning
from nilearn.image import resample_img
import nibabel as nib
import warnings
import numpy as np
from matplotlib import pyplot as plt

def load_data(external=False):
    if external:
        rest_dataset = datasets.fetch_development_fmri(n_subjects=20)
        func_filenames = rest_dataset.func
        confounds = rest_dataset.confounds
    else:
        # func_filenames = sorted(glob('data/functional/*.nii'))
        func_filenames = sorted(glob('data/resampled/*.nii'))
        confounds = [None] * len(func_filenames)                # Replace with actual confounds if any
        print(f"Found {len(func_filenames)} fMRI files") 

        # reference_img = nib.load(func_filenames[0])  # Use the first image as a reference
        # resampled_files = []
        # print(f"Image affine is {reference_img.affine}")

        # for filename in func_filenames:
        #     img = nib.load(filename)
            # print(f"Affine is {img.affine}")
            # resampled_img = resample_img(img, target_affine=reference_img.affine, target_shape=reference_img.shape[:3])
            # output_filename = f"resampled_{os.path.basename(filename)}"
            # resampled_img.to_filename(output_filename)
            # resampled_files.append(output_filename)

        # func_filenames = resampled_files  # Update file list with resampled images

    return func_filenames, confounds

def create_brain_maps(func_filenames):
    # DictLearning used to extract functional networks
    dict_learn = DictLearning(
        n_components=8,
        smoothing_fwhm=6.0,
        memory="nilearn_cache",
        memory_level=2,
        random_state=0,
        standardize="zscore_sample",
    )

    # Fit to the data
    dict_learn.fit(func_filenames)
    # Resting state networks/maps in attribute `components_img_`
    components_img = dict_learn.components_img_

    return dict_learn, components_img

def region_extraction(components_img):
    
    extractor = RegionExtractor(
        components_img,
        threshold=0.5,
        thresholding_strategy="ratio_n_voxels",
        extractor="local_regions",
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        min_region_size=1350,
    )
    # Just call fit() to process for regions extraction
    extractor.fit()
    # Extracted regions are stored in regions_img_
    regions_extracted_img = extractor.regions_img_
    # Each region index is stored in index_
    regions_index = extractor.index_
    # Total number of regions extracted
    n_regions_extracted = regions_extracted_img.shape[-1]

    return extractor, regions_extracted_img, n_regions_extracted, regions_index

def connectivity_analysis(extractor, func_filenames, confounds):
    
    correlations = []
    ## Initializing ConnectivityMeasure object with kind='correlation'
    ## Other measures exists in nilearn such as “correlation”, “partial correlation”, “tangent”, “covariance”, “precision”. 
    connectome_measure = connectome.ConnectivityMeasure(kind="correlation", standardize="zscore_sample")

    for filename, confound in zip(func_filenames, confounds):
        # call transform from RegionExtractor object to extract timeseries signals
        timeseries_each_subject = extractor.transform(filename, confounds=confound)
        # call fit_transform from ConnectivityMeasure object
        correlation = connectome_measure.fit_transform([timeseries_each_subject])
        # saving each subject correlation to correlations
        correlations.append(correlation)

    mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted, n_regions_extracted)

    return mean_correlations

# Main execution
if __name__ == "__main__":

    # Load data and create brain maps
    func_filenames, confounds = load_data(False)                        ## Which affine function to use?
    dict_learn, components_img = create_brain_maps(func_filenames)      ## Find functional networks, showing regions with correlated activity
    
    # Display functional networks 
    warnings.filterwarnings("ignore", category=UserWarning, message="linewidths is ignored by contourf")
    plotting.plot_prob_atlas(components_img, view_type="filled_contours", title="Functional networks (components) extracted by DictLearning")
    plt.savefig("functional_networks.png")
    plt.close()
    # plotting.show()
    
    # Extract brain regions from components and display them
    extractor, regions_extracted_img, n_regions_extracted, regions_index = region_extraction(components_img)
    title = (f"{n_regions_extracted} regions are extracted from 8 components.\nEach separate color of region indicates extracted region")
    plotting.plot_prob_atlas(regions_extracted_img, view_type="filled_contours", title=title)
    plt.savefig("regions_extracted.png")
    plt.close()
    # plotting.show()

    # Computing functional connectivity matrices
    mean_correlations = connectivity_analysis(extractor, func_filenames, confounds)
    title = f"Correlation between {int(n_regions_extracted)} regions"
    plotting.plot_matrix(mean_correlations, vmax=1, vmin=-1, colorbar=True, title=title)
    plt.savefig("connectome.png")
    plt.close()
    # plotting.show()
    
    # Display the connectome
    display = plotting.plot_matrix(mean_correlations, vmax=1, vmin=-1, colorbar=True, title=title)
    regions_img = regions_extracted_img
    coords_connectome = plotting.find_probabilistic_atlas_cut_coords(regions_img)
    plotting.plot_connectome(mean_correlations, coords_connectome, edge_threshold="90%", title=title)
    plt.savefig("connectome_display.png")
    plt.close()
    # plotting.show()

    # # Validating the results
    # img = image.index_img(components_img, 1)
    # coords = plotting.find_xyz_cut_coords(img)
    # display = plotting.plot_stat_map(img, cut_coords=coords, colorbar=False, title="Showing one specific network (component)")
    # plotting.show()

    # regions_indices_of_map3 = np.where(np.array(regions_index) == 1)
    # display = plotting.plot_anat(cut_coords=coords, title="Regions from this network")
    # # Add as an overlay all the regions of index 1
    # colors = "rgbcmyk"
    # for each_index_of_map3, color in zip(regions_indices_of_map3[0], colors):
    #     display.add_overlay(image.index_img(regions_extracted_img, each_index_of_map3), cmap=plotting.cm.alpha_cmap(color))
    # plotting.show()