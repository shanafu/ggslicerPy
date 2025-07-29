This repository is for the 'ggslicer' Python package.

Scientific visualization is an integral aspect to sharing research data. In the field of data science, data is commonly visualized with the ‘ggplot2’ package in R. In python, an equivalent ‘plotnine’ package is available for use. ‘ggslicer’ is a package available for both R and python that efficiently wrangles imaging data into a data frame to be utilized by ‘ggplot2’ and ‘plotnine’ for image visualization.

Built around the ‘SimpleITK’ image processing framework, ‘ggslicer’ is able to process 2D and 3D images such as DICOM, Nifti, MINC, and Analyze.

**How to get started:**

There are currently four functions in the package; two of which may be used to extract a data frame, and two that fix and preserve MINC image orientation when reading and writing the image with ‘SimpleITK’. The four functions are of the following:

**slice_axis() :** Read an image file and return a data frame containing voxel and world coordinates along specified slice.

**slice_intensity() :** Read an image file and optional data frame with specific coordinates and return a data frame with intensity values appended to each coordinate.

**ReadImage_fix() :** Read an image file with ‘SimpleITK’.

**WriteImage_fix() :** Write an image file with ‘SimpleITK’.
