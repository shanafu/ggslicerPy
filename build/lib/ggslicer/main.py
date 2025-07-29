import numpy
import pandas
from pathlib import Path
import SimpleITK
from tqdm import tqdm


def orientation_correction(image):
    """
    Internal function to read and fix orientation of MINC images

    Parameters
    ----------
    image : SimpleITK.Image
        An image object.

    Returns
    -------
    flipped_image : SimpleITK.Image
        Fixed image object.

    """
    flip_filter = SimpleITK.FlipImageFilter()

    # Flip along axes 0 and 1 (first two dimensions)
    flip_axes = [True, True, False]  # Flip X and Y, not Z
    flip_filter.SetFlipAxes(flip_axes)

    # Pass image through flip_filter
    flipped_image = flip_filter.Execute(image)

    # Copy metadata
    if image.HasMetaDataKey("OriginalFileType"):
        flipped_image.SetMetaData("OriginalFileType", image.GetMetaData("OriginalFileType"))
    return flipped_image


def ReadImage_fix(file):
    """
    Read an image and when MINC image, flip image orientation while preserving all image data

    Parameters
    ----------
    file : str or pathlib.Path
        Path to an image file.

    Returns
    -------
    image : SimpleITK.Image
        An image object or corrected image object.

    """
    image = SimpleITK.ReadImage(file)
    extension = Path(file).suffix.lower()

    is_minc = extension in [".mnc", ".minc"]

    if is_minc:
        image.SetMetaData("OriginalFileType", "MINC")

        # Store original MINC direction matrix
        orig_direction = image.GetDirection()
        image.SetMetaData("OriginalDirection", ",".join(map(str, orig_direction)))

        # Apply orientation correction for proper display
        corrected_image = orientation_correction(image)

        # Copy metadata to corrected image
        corrected_image.SetMetaData("OriginalFileType", "MINC")
        corrected_image.SetMetaData("OriginalDirection", ",".join(map(str, orig_direction)))

        image = corrected_image

        return image

    else:
        image.SetMetaData("OriginalFileType", "Other")

        return image


def WriteImage_fix(image, output_file):
    """
    Write out an image file and when MINC image, flip image orientation while preserving all image data

    Parameters
    ----------
    image : SimpleITK.Image
        An image object from ReadImage() or ReadImage_fix().
    output_file : str
        Name of export file.

    Returns
    -------
    None.

    """
    # Check metadata
    original_file_type_meta = image.HasMetaDataKey("OriginalFileType")
    was_original_minc = False

    if original_file_type_meta:
        original_type = image.GetMetaData("OriginalFileType")
        was_original_minc = original_type == "MINC"

    output_extension = Path(output_file).suffix.lower()
    is_output_minc = output_extension in [".mnc", ".minc"]
    is_output_nifti = output_extension in [".nii"]

    image_to_write = image

    if was_original_minc and is_output_minc:
        # MINC -> MINC: Undo the reading correction to restore proper MINC orientation
        image_to_write = orientation_correction(image)

    elif was_original_minc and is_output_nifti:
        # MINC -> non-MINC: Undo the reading correction to restore proper MINC orientation
        image_to_write = SimpleITK.Image(image)  # Copy the image
        ras_direction = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        image_to_write.SetDirection(ras_direction)

    SimpleITK.WriteImage(image_to_write, output_file)


def slice_axis(file, slice_orientation, slice_orientation_coords):
    """
    Extract voxel and world coordinates along specified slice

    Parameters
    ----------
    file : str or pathlib.Path
        Path to an image file.
    slice_orientation : str
        The desired axis to slice along. Options: "x"/"sagittal"/"1", "y"/"coronal"/"2", "z"/"axial"/"3".
    slice_orientation_coords : list or numpy.ndarray
        Numeric vector of voxel coordinates specifying which slice planes to extract along the chosen axis.

    Returns
    -------
    df_slice_axis : pandas.DataFrame
        A data frame with columns: x, y, z (world coordinates) and i, j, k (voxel coordinates) for all voxels on the specified slice planes.

    """
    image = ReadImage_fix(file)
    file_dims = image.GetSize()

    # Creates all possible voxel combinations
    i, j, k = numpy.meshgrid(
        numpy.arange(file_dims[0]),
        numpy.arange(file_dims[1]),
        numpy.arange(file_dims[2]),
        indexing='ij'

    )

    voxel_coords = pandas.DataFrame({
        'i': i.flatten(),
        'j': j.flatten(),
        'k': k.flatten()
    })

    # Convert voxel coordinates to world coordinates using the correct image
    world_coords_matrix = numpy.array([
        image.TransformIndexToPhysicalPoint([int(row[0]), int(row[1]), int(row[2])])
        for row in tqdm(voxel_coords[['i', 'j', 'k']].values)
    ])

    # Add world coordinates as columns
    voxel_coords['x'] = world_coords_matrix[0, :]
    voxel_coords['y'] = world_coords_matrix[1, :]
    voxel_coords['z'] = world_coords_matrix[2, :]

    axis_choice = slice_orientation.lower()
    filter_col_world = None

    if axis_choice in ["x", "sagittal", "1"]:
        filter_col_world = "x"

    elif axis_choice in ["y", "coronal", "2"]:
        filter_col_world = "y"

    elif axis_choice in ["z", "axial", "3"]:
        filter_col_world = "z"

    else:
        print("Invalid 'slice_orientation': '", slice_orientation,
              "'. Please choose 'x', 'sagittal', '1'; 'y', 'coronal', '2'; or 'z', 'axial', '3'.")

        return None

    # Get unique world coordinates for the chosen axis
    available_world_coords = voxel_coords[filter_col_world].unique()

    # Find the closest available coordinates to user input
    target_slices = [available_world_coords[numpy.abs(available_world_coords - target_coord).argmin()]
                     for target_coord in slice_orientation_coords]

    target_slices = numpy.unique(target_slices)

    # Filter coordinates based on target slices
    tolerance = 1e-6

    # Create boolean mask
    mask = voxel_coords[filter_col_world].apply(
        lambda coord: numpy.any(numpy.abs(coord - target_slices) < tolerance)
    )

    # Filter and sort
    df_slice_axis = voxel_coords[mask].sort_values(filter_col_world)

    # Reorder columns
    df_slice_axis = df_slice_axis[["x", "y", "z", "i", "j", "k"]]

    return df_slice_axis


def slice_intensity(file, df_slice_axis=None):
    """
    Append intensity values to slice_axis data frame

    Parameters
    ----------
    file : str or pathlib.Path
        Path to an image file.
    df_slice_axis : pandas.DataFrame, optional
        Data frame containing voxel coordinates with columns: i, j, k (voxel indices) and x, y, z (world coordinates). Typically output from slice_axis() function. The default is None.

    Returns
    -------
    df : pandas.DataFrame
        A data frame with columns: intensity, x, y, z, i, j, k.

    """
    image = ReadImage_fix(file)
    arr = numpy.array(image)

    # Extract intensity values
    intensity_values = numpy.zeros(len(df_slice_axis))

    for row_index in range(len(df_slice_axis)):
        i_coord = df_slice_axis['i'].iloc[row_index]
        j_coord = df_slice_axis['j'].iloc[row_index]
        k_coord = df_slice_axis['k'].iloc[row_index]

        # Check bounds
        if (i_coord < arr.shape[0] and j_coord < arr.shape[1] and k_coord < arr.shape[
            2] and i_coord >= 0 and j_coord >= 0 and k_coord >= 0):
            intensity_values[row_index] = arr[i_coord, j_coord, k_coord]

        else:
            intensity_values[row_index] = numpy.nan

    df_slice_axis['intensity'] = intensity_values

    # Return properly ordered dataframe
    df = df_slice_axis[["intensity", "x", "y", "z", "i", "k", "j"]]

    return df


