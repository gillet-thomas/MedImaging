import os
import cv2
import pydicom
import dicom2nifti
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# Convert DICOM to NIfTI
dicom_directory = '/mnt/data/iai/datasets/ADNI_rsFMRI/ADNI/002_S_0295/Resting_State_fMRI/2011-06-02_07_56_36.0/I238623'
output_file = 'data/nifti_from_dicom.nii'
dicom2nifti.convert_directory(dicom_directory, os.path.dirname(output_file))

# DICOM file loading
file = '/mnt/data/iai/datasets/ADNI_rsFMRI/ADNI/002_S_0295/Resting_State_fMRI/2011-06-02_07_56_36.0/I238623/ADNI_002_S_0295_MR_Resting_State_fMRI_br_raw_20110602125224332_1459_S110474_I238623.dcm'
dicom = pydicom.dcmread(file)


# Print the full DICOM metadata (header)
# print(dicom) 


# Extract and print key metadata (Direct access to DICOM tags)
print("Patient Name:", dicom.PatientName)                   # Direct access to DICOM tags
print("Patient Name:", dicom.get("PatientName"))            # Safe access to DICOM tags
print("Patient Name:", dicom.get((0x0010, 0x0010)).value)   # Access by tag number
# print("Patient ID:", dicom.PatientID)
# print("Study Date:", dicom.StudyDate)
# print("Modality:", dicom.Modality)
# print("Image Size:", dicom.Rows, "x", dicom.Columns)
print("Pixel Spacing:", dicom.PixelSpacing if 'PixelSpacing' in dicom else "Not available")
print("Pixel Spacing:", dicom.get("PixelSpacing", "Not available"))


# # Visualize DICOM image
# pixel_array = dicom.pixel_array       # Converts to NumPy array
# plt.imshow(pixel_array, cmap="gray")  # Use gray colormap


# # Normalize to 8-bit (0-255 range)
# image_8bit = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# cv2.imwrite("dicom_image.png", image_8bit)