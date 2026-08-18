import sys
import types
from pathlib import Path

import nexusformat.nexus as nxs
import numpy as np
import pytest

from mdx2.data import ImageSeries
from mdx2.io import loadobj, nxsave, saveobj

# Create a test module with a class that lacks from_nexus method for security testing
_test_module = types.ModuleType("mdx2.test_no_from_nexus")
_test_module.ClassWithoutFromNexus = type("ClassWithoutFromNexus", (), {})
sys.modules["mdx2.test_no_from_nexus"] = _test_module


def test_writing_data_after_nxsave(tmp_path):
    """check that data can be written after saving a Nexus object"""

    # write a nexus file with an empty NXfield
    filename = tmp_path / "test_data_after_nxsave.nxs"
    nxfield = nxs.NXfield(shape=(10,), dtype="int32")
    nxdata = nxs.NXdata(nxfield)
    nxsave(nxdata, filename)
    # now, write some data to the NXfield
    nxfield[0] = 42

    # reload the file and check that the data was written correctly
    nxroot = nxs.nxload(filename)
    assert nxroot.entry.data.signal.nxdata[0] == 42


def test_ImageSeries_virtual_dataset_creation(tmp_path):
    """check that ImageSeries can be saved with a virtual dataset"""
    data_opts = {"dtype": np.int32, "compression": "gzip", "compression_opts": 1, "shuffle": True}

    phi = np.arange(20)
    iy = np.arange(10)
    ix = np.arange(10)
    data = nxs.NXfield(shape=(20, 10, 10), name="data", **data_opts, chunks=(8, 5, 5))
    exposure_times = 0.1 * np.ones_like(phi)

    image_series = ImageSeries(phi, iy, ix, data, exposure_times)
    filename = tmp_path / "test_image_series_virtual.nxs"
    nxobj = image_series.save(filename, virtual=True)
    assert isinstance(nxobj.data, nxs.NXvirtualfield)
    assert isinstance(image_series.data, nxs.NXvirtualfield)
    assert image_series.chunks == (8, 5, 5)
    assert image_series.data.shape == (20, 10, 10)
    assert len(nxobj.data._vfiles) == 3  # source files are present

    # write some data to the second source file, check that it appears in the virtual dataset
    source_series = ImageSeries.load(nxobj.data._vfiles[1], mode="r+")
    source_series.data[0, 0, 0] = 666
    assert image_series.data[8, 0, 0] == 666


def test_ImageSeries_virtual_dataset_creation_nested_paths(tmp_path):
    """check source_directory is not duplicated when outfile has its own subdirectories (regression test)"""
    data_opts = {"dtype": np.int32, "compression": "gzip", "compression_opts": 1, "shuffle": True}

    phi = np.arange(20)
    iy = np.arange(10)
    ix = np.arange(10)
    data = nxs.NXfield(shape=(20, 10, 10), name="data", **data_opts, chunks=(8, 5, 5))
    exposure_times = 0.1 * np.ones_like(phi)

    # outfile and datastore share a common parent subdirectory, e.g. "results/exp1"
    outdir = tmp_path / "results" / "exp1"
    filename = outdir / "data.nxs"
    source_directory = outdir / "datastore"

    image_series = ImageSeries(phi, iy, ix, data, exposure_times)
    nxobj = image_series.save(filename, virtual=True, source_directory=str(source_directory))

    assert len(nxobj.data._vfiles) == 3
    for vfile in nxobj.data._vfiles:
        # source files should live directly under source_directory, not in a duplicated nested subdirectory
        assert Path(vfile).parent == source_directory
        assert Path(vfile).is_file()

    # reloading from disk should also resolve the virtual sources correctly
    reloaded = ImageSeries.load(filename)
    assert reloaded.virtual_source_files == nxobj.data._vfiles
    assert reloaded.data.shape == (20, 10, 10)


def test_loadobj_security_checks(tmp_path):
    """Test that loadobj rejects untrusted modules and validates interface."""
    # Create a simple mdx2 object and save it properly
    phi = np.arange(10)
    iy = np.arange(5)
    ix = np.arange(5)
    data = nxs.NXfield(shape=(10, 5, 5), dtype=np.int32)
    exposure_times = 0.1 * np.ones_like(phi)
    image_series = ImageSeries(phi, iy, ix, data, exposure_times)

    filename = tmp_path / "test_security.nxs"
    saveobj(image_series, filename, name="test_obj")

    # Test 1: Attempt to load with a malicious module name
    nxroot = nxs.nxload(filename, "r+")
    nxroot.entry.test_obj.attrs["mdx2_module"] = "os"  # Malicious module
    nxroot.entry.test_obj.attrs["mdx2_class"] = "system"
    nxroot.save(filename, mode="w")

    with pytest.raises(ValueError, match=r"Untrusted module 'os': only mdx2\.\* modules are allowed"):
        loadobj(filename, "test_obj")

    # Test 2: Attempt to load with a non-existent class
    nxroot = nxs.nxload(filename, "r+")
    nxroot.entry.test_obj.attrs["mdx2_module"] = "mdx2.data"
    nxroot.entry.test_obj.attrs["mdx2_class"] = "NonExistentClass"
    nxroot.save(filename, mode="w")

    with pytest.raises(AttributeError):  # getattr will raise AttributeError
        loadobj(filename, "test_obj")

    # Test 3: Attempt to load with a class that exists but lacks from_nexus method
    nxroot = nxs.nxload(filename, "r+")
    nxroot.entry.test_obj.attrs["mdx2_module"] = "mdx2.test_no_from_nexus"
    nxroot.entry.test_obj.attrs["mdx2_class"] = "ClassWithoutFromNexus"
    nxroot.save(filename, mode="w")

    with pytest.raises(TypeError, match=r"does not have a callable from_nexus method"):
        loadobj(filename, "test_obj")
