import pytest
import SimpleITK
import numpy
from pathlib import Path

from ggslicer.readwrite import orientation_correction, ReadImage_fix, WriteImage_fix

# Fixtures for image test files
MNC_FILE = "test.mnc"
NII_FILE = "test.nii"

@pytest.fixture
def mnc_image():
    return SimpleITK.ReadImage(MNC_FILE)

@pytest.fixture
def nii_image():
    return SimpleITK.ReadImage(NII_FILE)

# orientation_correction; tests flip function
class Testorientation_correction:

    # Flips image twice to check if output returns original
    def test_doubleflipimage_mnc(self, mnc_image):
        flipped = orientation_correction(mnc_image)
        double_flipped = orientation_correction(flipped)
        numpy.testing.assert_array_equal(
            SimpleITK.GetArrayFromImage(mnc_image),
            SimpleITK.GetArrayFromImage(double_flipped),
        )

    # Checks X and Y are mirrored, but Z is unchanged
    def test_z_originalaxis_mnc(self, mnc_image):
        arr = SimpleITK.GetArrayFromImage(mnc_image)
        flipped_arr = SimpleITK.GetArrayFromImage(orientation_correction(mnc_image))
        numpy.testing.assert_array_equal(arr[0], flipped_arr[0, ::-1, ::-1])

    # Preserve original metadata as image is flipped
    def test_origmetadata(self, mnc_image):
        mnc_image.SetMetaData("OriginalFileType", "MINC")
        flipped = orientation_correction(mnc_image)
        assert flipped.GetMetaData("OriginalFileType") == "MINC"


# ReadImage_fix; test labels and storage of data
class TestReadImage_fix:

    # Reads extension .mnc or .minc and gets tagged as MINC
    def test_minc_correcttag(self):
        result = ReadImage_fix(MNC_FILE)
        assert result.GetMetaData("OriginalFileType") == "MINC"

    # Checks morginal 3x3 direction matrix was saved as metadata
    def test_minc_storeorigdirection(self):
        result = ReadImage_fix(MNC_FILE)
        assert result.HasMetaDataKey("OriginalDirection")
        values = [float(v) for v in result.GetMetaData("OriginalDirection").split(",")] # 3x3 matrix
        assert len(values) == 9

    # Reads extension .nii and gets tagged as Other
    def test_nifti_correcttag(self):
        result = ReadImage_fix(NII_FILE)
        assert result.GetMetaData("OriginalFileType") == "Other"

    # Reads .nii file with ReadImage_fix and SimpleITK and confirm data is unchanged
    def test_nifti_origdata(self):
        original = SimpleITK.ReadImage(NII_FILE)
        result = ReadImage_fix(NII_FILE)
        numpy.testing.assert_array_equal(
            SimpleITK.GetArrayFromImage(original),
            SimpleITK.GetArrayFromImage(result),
        )

# WriteImage_fix; tests for correct image output
class TestWriteImage_fix:
    
    # MINC -> MINC; ensures data matches with original and flip is undone from Read
    def test_minc_to_minc_restoreorientation(self, tmp_path):
        original = SimpleITK.ReadImage(MNC_FILE)
        out = str(tmp_path / "out.mnc")

        read_img = ReadImage_fix(MNC_FILE)
        WriteImage_fix(read_img, out)

        reread = SimpleITK.ReadImage(out)
        numpy.testing.assert_array_almost_equal(
            SimpleITK.GetArrayFromImage(original),
            SimpleITK.GetArrayFromImage(reread),
            decimal = 4, # Wiggle room for rounding errors w/in 4 decimal places
        )

    # MINC -> NIFTI; checks direction matrix is like RAS matrix
    def test_minc_to_nifti_rasdirection(self, tmp_path):
        out = str(tmp_path / "out.nii")
        read_img = ReadImage_fix(MNC_FILE)
        WriteImage_fix(read_img, out)

        reread = SimpleITK.ReadImage(out)
        expected = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) # RAS direction matrix
        assert reread.GetDirection() == pytest.approx(expected, abs = 1e-5) # Floating point comparison taking account rounding errors

    # NIFTI -> NIFTI; check data is unchanged
    def test_nifti_to_nifti_origdata(self, tmp_path):
        original = SimpleITK.ReadImage(NII_FILE)
        out = str(tmp_path / "out.nii")

        read_img = ReadImage_fix(NII_FILE)
        WriteImage_fix(read_img, out)

        reread = SimpleITK.ReadImage(out)
        numpy.testing.assert_array_equal(
            SimpleITK.GetArrayFromImage(original),
            SimpleITK.GetArrayFromImage(reread),
        )

    # NIFTI -> MINC; 
    def test_nifti_to_minc():
