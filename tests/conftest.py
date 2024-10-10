# -*- coding: utf-8 -*-
import pytest
from pathlib import Path

"""
    Dummy conftest.py for qa4sm_reader.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

def pytest_collection_modifyitems(items):
    # Move the test_utils::test_transcribe to the beginning of the list, as it is required for transcribing test files for other tests
    first_test = None
    for item in items:
        if item.name == "test_transcribe":
            first_test = item
            break
    if first_test:
        items.insert(0, items.pop(items.index(first_test)))


@pytest.fixture(scope="session")
def TEST_DATA_DIR():
    return Path(__file__).parent / 'test_data'
