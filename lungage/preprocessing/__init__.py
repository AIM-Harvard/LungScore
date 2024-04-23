from .lung_preprocessing import lung_extraction
from .lung_preprocessing import resample_and_resize
from lungmask import mask

#preprocess nrrd
def preprocess(nrrd_path):

    nrrd_scan = preprocess(nrrd_path) 

    return nrrd_scan

#segment the lung from nrrd using lung mask model
def segment_lung(nrrd_scan):

    segmented_lung = mask.apply(nrrd_scan)

    return segmented_lung

#extract and preprocess lung from segmented lung
def extract_lung(lungmask, nrrd):

    extracted_lung = lung_extraction(lungmask, nrrd)

    return extracted_lung