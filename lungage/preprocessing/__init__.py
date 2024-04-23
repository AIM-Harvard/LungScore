from .dcm_to_nrrds import dicom_to_nrrd
from .lung_preprocessing import lung_preprocess

def run_core(folder_to_dcms):

    nrrd_scan = dicom_to_nrrd(folder_to_dcms) 

    return nrrd_scan

def extract_lung(nrrd_scan):

    extracted_lung = lung_preprocess(nrrd_scan)

    return extracted_lung