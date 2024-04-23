"""
  Deep-learning biomarker for Lung Health - resampling and resizing NRRD scan
"""

import os
import numpy as np
import SimpleITK as sitk


def resample_and_resize(image_path, new_spacing=[0.68, 0.68, 2.5]):
    # Load the NRRD image
    image = sitk.ReadImage(image_path)

    orig_size = image.GetSize()
    orig_spacing = image.GetSpacing()

    if orig_spacing > 3.27:
        print("Spacing out of range, spacing should be less than or equal 3.27")
        quit()     

    new_size = [int(orig_size[0] * orig_spacing[0] / new_spacing[0]),
                int(orig_size[1] * orig_spacing[1] / new_spacing[1]),
                int(orig_size[2] * orig_spacing[2] / new_spacing[2])]

    # Resample the image to the new spacing
    original_spacing = image.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)

    resampler.SetSize(new_size)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputPixelType(image.GetPixelID())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())

    resampled_image = resampler.Execute(image)
    
    resampled_image_size = resampled_image.GetSize()

    final_size = [512, 512, 0]

    old_size = resampled_image.GetSize()


    new_size_down = [max(0, int((old_size[0] - final_size[0]) / 2)),
                    max(0, int((old_size[1] - final_size[1]) / 2)),
                    0]
    new_size_up = [max(0, old_size[0] - final_size[0] - new_size_down[0]),
                    max(0, old_size[1] - final_size[1] - new_size_down[1]),
                    0]
    
    crop_filter = sitk.CropImageFilter()
    crop_filter.SetUpperBoundaryCropSize(new_size_up)
    crop_filter.SetLowerBoundaryCropSize(new_size_down)
    resized_image = crop_filter.Execute(resampled_image)


    old_size = resized_image.GetSize()

    new_size_down = [max(0, int((final_size[0] - old_size[0]) / 2)),
                    max(0, int((final_size[1] - old_size[1]) / 2)),
                    0]
    new_size_up = [max(0, final_size[0] - old_size[0] - new_size_down[0]),
                    max(0, final_size[1] - old_size[1] - new_size_down[1]),
                    0]

    air = -1024
    pad_filter = sitk.ConstantPadImageFilter()
    pad_filter.SetConstant(air)
    pad_filter.SetPadUpperBound(new_size_up)
    pad_filter.SetPadLowerBound(new_size_down)
    resized_image = pad_filter.Execute(resized_image)

    return resized_image
