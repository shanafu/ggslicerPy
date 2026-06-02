from ggslicer.readwrite import ReadImage_fix, WriteImage_fix
import os
from pathlib import Path

import SimpleITK

# MINC -> NIFTI
image = ReadImage_fix("test.mnc")
WriteImage_fix(image, "output_mn.nii")


# MINC -> MINC
image = ReadImage_fix("test.mnc")
WriteImage_fix(image, "output_mm.mnc")


# NIFTI -> NIFTI
image = ReadImage_fix("test.nii")
WriteImage_fix(image, "output_nn.nii")

# NIFTI -> MINC
image = ReadImage_fix("test.nii")
WriteImage_fix(image, "output_nm.mnc")
