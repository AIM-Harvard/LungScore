from .lung_preprocessing import lung_extraction
from .lung_preprocessing import resample_and_resize
from .lung_preprocessing import dicom_to_nrrd
from lungmask import mask

#DICOM to NRRD
def dcm_to_nrrd(dcm_path):

    nrrd_scan = dicom_to_nrrd(dcm_path) 

    return nrrd_scan

#preprocess nrrd
def preprocess(nrrd_path):

    nrrd_scan = resample_and_resize(nrrd_path) 

    return nrrd_scan

#segment the lung from nrrd using lung mask model
def segment_lung(nrrd_scan):

    segmented_lung = mask.apply(nrrd_scan)

    return segmented_lung

#extract and preprocess lung from segmented lung
def extract_lung(lungmask, nrrd):

    extracted_lung = lung_extraction(lungmask, nrrd)

    return extracted_lung