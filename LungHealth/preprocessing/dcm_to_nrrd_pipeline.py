"""
  Deep-learning biomarker for Lung Health DICOM to NRRD
"""
import yaml
import argparse
import matplotlib  

import sys, os, glob, socket, glob, csv
import pydicom
import numpy as np
import multiprocessing
import SimpleITK as sitk
from functools import partial
from multiprocessing import Process, Manager
import pandas as pd


base_conf_file_path = 'config/'
conf_file_list = [f for f in os.listdir(base_conf_file_path) if f.split('.')[-1] == 'yaml']

parser = argparse.ArgumentParser(description = 'DCM to NRRD pipeline')

parser.add_argument('--conf',
                    required = False,
                    help = 'Specify the YAML configuration file containing the preprocessing details. ' \
                            + 'Defaults to "dcm_to_nrrd_pipeline.yaml"',
                    choices = conf_file_list,
                    default = "dcm_to_nrrd_pipeline.yaml",
                   )


args = parser.parse_args()

conf_file_path = os.path.join(base_conf_file_path, args.conf)

with open(conf_file_path) as f:
  yaml_conf = yaml.load(f, Loader = yaml.FullLoader)

# input-output Paths
DICOM_folder_path = yaml_conf["io"]["DICOM_folder_path"]
NRRD_folder_path = yaml_conf["io"]["NRRD_folder_path"]

# Processing settings
NUM_CORES = yaml_conf["preprocessing"]["NUM_CORES"]
OVERWRITE = yaml_conf["preprocessing"]["OVERWRITE"] 

# Selection settings
MIN_SLICES = yaml_conf["selection"]["MIN_SLICES"]
MAX_THICKNESS = yaml_conf["selection"]["MAX_THICKNESS"] 
MIN_THICKNESS = yaml_conf["selection"]["MIN_THICKNESS"] 

CURATED_SPACING = yaml_conf["selection"]["CURATED_SPACING"]
CURATED_SIZE = yaml_conf["selection"]["CURATED_SIZE"] 


# --------------------------------------------------------------------------------------------------
def get_sitk_image(subject_id, dicom_data):
  img_cube = np.asarray([dicom_slice.pixel_array for dicom_slice in dicom_data], dtype=np.int16)

  # Calculate slice spacing
  slice_spacing = np.abs(dicom_data[1].ImagePositionPatient[2] -
                         dicom_data[2].ImagePositionPatient[2])
  
  img_cube[img_cube==-2000] = 0
  img_cube *= np.int16(dicom_data[0].RescaleSlope)
  img_cube += np.int16(dicom_data[0].RescaleIntercept)

  img_sitk = sitk.GetImageFromArray(img_cube)
  img_spacing = [float(dicom_data[0].PixelSpacing[0]), float(dicom_data[0].PixelSpacing[1]),
                 slice_spacing]

  if 0.0 in img_spacing:
    print('ERROR - Zero spacing found for patient,', subject_id, img_spacing)
    raise Exception('Exception')
    return

  if img_spacing < MIN_THICKNESS or img_spacing > MAX_THICKNESS:
    print('ERROR - out of range spacing found for patient,', subject_id, img_spacing)
    raise Exception('Exception')
    return

  img_sitk.SetSpacing(img_spacing)
  img_direction = [int(i) for i in dicom_data[0].ImageOrientationPatient] + \
                  [0, 0, 1]  # Add third dimension
  img_sitk.SetDirection(img_direction)
  img_origin = dicom_data[0].ImagePositionPatient
  img_sitk.SetOrigin(img_origin)

  return img_sitk


# --------------------------------------------------------------------------------------------------
def resample_sitk(cube_sitk, method, spacing):
  orig_size = cube_sitk.GetSize()
  orig_spacing = cube_sitk.GetSpacing()

  new_size = [int(orig_size[0] * orig_spacing[0] / spacing[0]),
              int(orig_size[1] * orig_spacing[1] / spacing[1]),
              int(orig_size[2] * orig_spacing[2] / spacing[2])]

  res_filter = sitk.ResampleImageFilter()
  res_filter.SetSize(new_size)
  res_filter.SetTransform(sitk.Transform())
  res_filter.SetInterpolator(method)
  res_filter.SetOutputOrigin(cube_sitk.GetOrigin())
  res_filter.SetOutputSpacing(spacing)
  res_filter.SetOutputDirection(cube_sitk.GetDirection())
  res_filter.SetOutputPixelType(cube_sitk.GetPixelID())
  res_filter.SetDefaultPixelValue(cube_sitk.GetPixelIDValue())
  cube_sitk = res_filter.Execute(cube_sitk)

  return cube_sitk


# --------------------------------------------------------------------------------------------------
# Resize volume to 512x512 in x and y but leave z untouched
def resize_sitk(cube_sitk, air, final_size):
  # 1) Crop down bigger axes
  old_size = cube_sitk.GetSize()

  new_size_down = [max(0, int((old_size[0] - final_size[0]) / 2)),
                   max(0, int((old_size[1] - final_size[1]) / 2)),
                   0]
  new_size_up = [max(0, old_size[0] - final_size[0] - new_size_down[0]),
                 max(0, old_size[1] - final_size[1] - new_size_down[1]),
                 0]

  crop_filter = sitk.CropImageFilter()
  crop_filter.SetUpperBoundaryCropSize(new_size_up)
  crop_filter.SetLowerBoundaryCropSize(new_size_down)
  cube_sitk = crop_filter.Execute(cube_sitk)

  # 2) Pad smaller axes
  old_size = cube_sitk.GetSize()

  new_size_down = [max(0, int((final_size[0] - old_size[0]) / 2)),
                   max(0, int((final_size[1] - old_size[1]) / 2)),
                   0]
  new_size_up = [max(0, final_size[0] - old_size[0] - new_size_down[0]),
                 max(0, final_size[1] - old_size[1] - new_size_down[1]),
                 0]

  pad_filter = sitk.ConstantPadImageFilter()
  pad_filter.SetConstant(air)
  pad_filter.SetPadUpperBound(new_size_up)
  pad_filter.SetPadLowerBound(new_size_down)
  cube_sitk = pad_filter.Execute(cube_sitk)

  return cube_sitk


# ----------------------------------------------------------------------------------------------------------------------
def run_core(dicom_path):
  nrrd_writer = sitk.ImageFileWriter()

  for fold in os.listdir(dicom_path):
    pat_folder = dicom_path+fold+'/'      
    pat_id = os.path.basename(os.path.normpath(pat_folder))

    print('Processing subject', pat_id)

    series_dir = pat_folder

    nrrd_file = os.path.join(NRRD_folder_path, pat_id + '.nrrd')

    try:
      dcm_files = glob.glob(series_dir+"/" + "/*.dcm")
      dicom_data = [pydicom.read_file(dcm_file, force=True) for dcm_file in dcm_files]
      dicom_data.sort(key=lambda x: float(x.ImagePositionPatient[2]))
      
      img_sitk = get_sitk_image(pat_id, dicom_data)
      
      img_sitk = resample_sitk(img_sitk, sitk.sitkLinear, CURATED_SPACING)
      if not tuple(np.round(CURATED_SPACING, 1)) == tuple(np.round(img_sitk.GetSpacing(), 1)):
        print('Wrong final spacing, %s, %s, %s' % (pat_id, CURATED_SPACING, str(img_sitk.GetSpacing())))
      
      # Resize image so all images are the same
      img_sitk = resize_sitk(img_sitk, -1024, CURATED_SIZE)
      if not tuple(CURATED_SIZE[0:2]) == img_sitk.GetSize()[0:2]:
        print('Wrong final size %s, %s, %s' % (pat_id, CURATED_SIZE, str(img_sitk.GetSize())))

      nrrd_writer.SetFileName(nrrd_file)
      nrrd_writer.SetUseCompression(True)
      nrrd_writer.Execute(img_sitk)
    except Exception as e:
          print(pat_id) 


if __name__ == "__main__":
 
  print("\nDICOMs TO NRRDs.. ---\n")
  run_core(DICOM_folder_path)
    
    