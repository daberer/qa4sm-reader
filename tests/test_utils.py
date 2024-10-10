import os
from pathlib import Path
from glob import glob
import xarray as xr

import qa4sm_reader
from qa4sm_reader.utils import transcribe
from qa4sm_reader.globals import TEMPORAL_SUB_WINDOW_NC_COORD_NAME

def test_get_version():
    assert qa4sm_reader.__version__ != 'unknown'

def test_transcribe_all_testfiles():
    # check if all test files can be transcribed for subsequent tests. proper testing of the transcription is done in test_netcdf_transcription.py
    TEST_FILE_ROOT = Path(Path(os.path.dirname(os.path.abspath(__file__))).parent, 'tests', 'test_data')
    test_files = [
        x for x in glob(str(TEST_FILE_ROOT / '**/*.nc'), recursive=True)
        if 'intra_annual' not in Path(x).parts
    ]   # ignore the dedicated intra-annual test files for now, as they will be tested separately in depth

    assert len(test_files) == 13

    assert any([isinstance(transcribe(f), xr.Dataset) for f in test_files])

    assert any([TEMPORAL_SUB_WINDOW_NC_COORD_NAME in transcribe(f).dims for f in test_files])
