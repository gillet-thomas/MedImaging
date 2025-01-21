import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def segmentation_mask(image):
    image = image.astype(np.uint16)  # Convert 8-bit to 16-bit
    
    # Apply Otsu's thresholding for segmentation
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure the segmented image is 8-bit (required by cv2.findContours)
    mask = mask.astype(np.uint8)  # Convert 16-bit to 8-bit

    # Find contours of the segmented image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill segmentation mask to outer contours
    cv2.drawContours(mask, [contours[0]], -1, (255), thickness=cv2.FILLED)  # -1 means draw all contours, (255) is the color

    return mask
    

if __name__ == "__main__":
    file = '/mnt/data/iai/datasets/ADNI_rsFMRI/ADNI/002_S_0295/Resting_State_fMRI/2011-06-02_07_56_36.0/I238623/ADNI_002_S_0295_MR_Resting_State_fMRI_br_raw_20110602125224332_1459_S110474_I238623.dcm'
    image = pydicom.dcmread(file).pixel_array       # Read the fmri pixel data, 64x64 shape
    plt.imshow(image)
    plt.savefig('brain_segmentation/dicom.png')

    # file = '/mnt/data/iai/Projects/ABCDE/fmris/fmris-viz/501_resting_state_fmri.nii.gz'
    # nifti = nib.load(file)
    # nifti_data = nifti.get_fdata()      # (64, 64, 48, 140)
    # image = nifti_data[:, :, 0, 0]      # Get the first slice of the 4D volume
    # plt.imshow(image)
    # plt.savefig('brain_segmentation/nifti.png')

    segmentation_mask = segmentation_mask(image)

    plt.imshow(segmentation_mask)
    plt.savefig('brain_segmentation/segmentation_mask.png')

    plt.imshow(image)
    plt.contour(segmentation_mask, colors='red', linewidths=0.5)
    plt.savefig('brain_segmentation/segmented_image.png')

