from .lung_preprocessing import lung_extraction
from .lung_preprocessing import resample_and_resize
#from .lung_preprocessing import dicom_to_nrrd
from lungmask import mask

#DICOM to NRRD
# def dcm_to_nrrd(dcm_path):
#     """
#     Convert DICOM to NRRD - given DICOM folder with all dcm files of the same scan
#     Args:
#         dcm_path (str): Folder with all DCM files of the same scan
#     Returns:
#         nrrd_scan: NRRD Scan
#     """
#     nrrd_scan = dicom_to_nrrd(dcm_path) 

#     return nrrd_scan


#preprocess nrrd
def preprocess(nrrd_path):
    """
    Read and Resample NRRD based on AI Lung Health selection
    Args:
        nrrd_path (str): path to NRRD Scan
    Returns:
        nrrd_scan: NRRD resampled Scan
    """
    nrrd_scan = resample_and_resize(nrrd_path) 

    return nrrd_scan

#segment the lung from nrrd using lung mask model
def segment_lung(nrrd_scan):
    """
    Segment the lung from loaded NRRD scan using - lung mask model -
    Args:
        nrrd_scan: nrrd scan in loaded mode
    Returns:
        segmented lung: output of lung mask model
    """
    segmented_lung = mask.apply(nrrd_scan)

    return segmented_lung

#extract and preprocess lung from segmented lung
def extract_lung(lungmask, nrrd):
    """
    Extract and preprocess lung mask model output
    Args:
       lungmask: the output of lung mask model
       nrrd: the corresponding loaded NRRD scan 
    Returns:
        extracted_lung: extracted and preprocessing lung 
    """
    extracted_lung = lung_extraction(lungmask, nrrd)

    return extracted_lung