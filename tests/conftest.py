import pytest
import mudata as md
import anndata as ad
import numpy as np
import pandas as pd
from typing import List

md.set_options(pull_on_update=False)

@pytest.fixture(scope="session") # Use session scope if this MuData object is large and doesn't change
def large_mdata_object():
    """Provides a large (but still testable) MuData object for multiple tests."""
    print("\nSetting up large_mdata_fixture (session scope)")

    bcs = [f'cell_{i}' for i in range(100)]
    genes = [f'gene_{j}' for j in range(50)]
    prots = [f'prot_{k}' for k in range(20)]

    # Simple object for now, eventually simulate a more complex MuData object
    adata_rna = ad.AnnData(np.random.rand(100, 50),
                           obs=pd.DataFrame(index=bcs),
                           var=pd.DataFrame(index=genes))
    adata_prot = ad.AnnData(np.random.rand(100, 20),
                            obs=pd.DataFrame(index=bcs),
                            var=pd.DataFrame(index=prots))
    mdata = md.MuData({'rna': adata_rna, 'adt': adata_prot})

    yield mdata

    print("\nTeardown large_mdata_fixture (session scope)") # Runs after all tests

# def assert_mudata_equal(mdata1: md.MuData, mdata2: md.MuData):
#     """
#     Custom assertion to check if two MuData objects are equivalent.
#     """
#     assert isinstance(mdata1, md.MuData)
#     assert isinstance(mdata2, md.MuData)
#     assert mdata1.n_obs == mdata2.n_obs
#     assert mdata1.n_vars == mdata2.n_vars
#     assert sorted(mdata1.mod.keys()) == sorted(mdata2.mod.keys())

#     for key in mdata1.mod:
#         # Simple check for AnnData equivalence within MuData
#         # In real life, you'd use AnnData's testing utilities or custom deep comparison
#         assert mdata1.mod[key].shape == mdata2.mod[key].shape
#         np.testing.assert_array_equal(mdata1.mod[key].X, mdata2.mod[key].X)
#         pd.testing.assert_frame_equal(mdata1.mod[key].obs, mdata2.mod[key].obs)
#         pd.testing.assert_frame_equal(mdata1.mod[key].var, mdata2.mod[key].var)
