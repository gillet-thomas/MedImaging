from nilearn import datasets, plotting
from nilearn.image import mean_img, load_img, index_img
import pandas as pd
from nilearn.decoding import Decoder
import numpy as np
from matplotlib import pyplot as plt

# Load Haxby dataset
haxby_dataset = datasets.fetch_haxby()

# Load anatomical image
anat_image = load_img(haxby_dataset.anat)   # Anatomical image
plotting.plot_anat(anat_image)              # Plot the anatomical image
plt.savefig("svm/anat_image.png")               # Save the anatomical image
plt.close()

# Load functional image
func_image = load_img(haxby_dataset.func[0])        # Functional image
func_image_mean = mean_img(func_image)              # Mean functional image
plotting.plot_epi(func_image_mean, cmap='magma')    # Plot the anatomical image
plt.savefig("svm/func_image.png")                       # Save the functional image
plt.close() 

# Load mask
vt_mask = load_img(haxby_dataset.mask_vt)                       # VT mask
plotting.plot_roi(vt_mask, bg_img=anat_image, cmap='Paired')    # Plot the VT mask
plt.savefig("svm/vt_mask.png")                                      # Save the VT mask
plt.close()

# if len(vt_mask.shape) == 4:
#     vt_mask_3d = index_img(vt_mask, 0)  # Extract the first volume

# Load behavioral data
mask_filepath = haxby_dataset.mask_vt[0]                     # Mask file path
fmri_filepath = haxby_dataset.func[0]                       # Functional images file path
csv_file = haxby_dataset.session_target[0]
behavioral_data = pd.read_csv(csv_file, delimiter=" ")      # [1452 rows x 2 columns]
labels = behavioral_data["labels"]
condition_mask = labels.isin(["face", "cat"])
fmri_niimgs = index_img(fmri_filepath, condition_mask)     
labels = labels[condition_mask]                             # [216 rows x 1 columns]

# Define Decoder object and fit the model
decoder = Decoder(estimator='svc', mask=mask_filepath, standardize="zscore_sample", cv=5, scoring='f1')
decoder.fit(fmri_niimgs, labels)
prediction = decoder.predict(fmri_niimgs)
# print(f"{prediction=}")

# Print F1 scores for each category
categories = np.unique(labels)
print('F1 scores:')
for category in categories:
    if category != 'rest':  # Exclude the "rest" category if desired
        print(f"{category.ljust(15)}    {np.mean(decoder.cv_scores_[category]):.2f}")

# Visualize SVM weights for a specific category (e.g., 'face')
category_to_visualize = 'face'
plotting.view_img(
    decoder.coef_img_[category_to_visualize], 
    bg_img=anat_image, 
    title=f"SVM weights for {category_to_visualize}", 
    dim=-1, 
    resampling_interpolation='nearest'
).open_in_browser() # change to plot_stat_map to save the image                      
# plt.savefig(f"svm/svm_weights_{category_to_visualize}.png")       
# plt.close()

