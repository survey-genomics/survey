import pytest

def test_mdata_modality_count(large_mdata_object):
    assert len(large_mdata_object.mod) == 2

def test_mdata_cell_names(large_mdata_object):
    assert 'cell_1' in large_mdata_object.obs_names
    assert 'cell_2' in large_mdata_object.obs_names