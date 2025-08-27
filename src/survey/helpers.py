# Built-ins
import re

# Standard libs
import pandas as pd

def get_chipset_params(meta_rxn):
    '''
    Extracts array parameters and chip metadata from a metadata pandas DataFrame.
    This DataFrame is typically created during Survey Genomics data processing,
    created from our Compiled Results spreadsheet. It has a specific structure
    and set of columns that are expected.
    '''

    shape_to_n = {
        'Square': 4,
        'Hexagon': 6,
    }

    array_params_cols_mapper = {
        'well-shape': 'n',
        'well-side-length': 's',
        'wall-width': 'w',
    }

    chip_meta_cols_mapper = {
        'chip-version': 'version',
        'chip-num': 'num',
        'layout-id': 'lyt',
        'layout-arr-offset': 'offset',
        'image_file': 'img',
    }

    def convert_offset(offset_string, match=False):
        """
        Validates the format of the offset string.
        Expected format: '(x, y)' where x and y are integers.
        """
        if match:
            return re.match(r'^\(\s*\d+\s*,\s*\d+\s*\)$', offset_string)
        else:
            return tuple([int(i.strip()) for i in offset_string.strip('()').split(',')])
        
    def get_array_params(meta_rxn):

        array_params_cols = ['chip-version', 'well-shape', 'well-side-length', 'wall-width', 'arr-num-rows', 'arr-num-cols']
        try:
            pre_array_params = meta_rxn[array_params_cols]
        except KeyError as e:
            raise KeyError(f"Meta DataFrame must contain the following columns: {array_params_cols}. Missing column: {e.args[0]}")
        try:
            pre_array_params = pre_array_params.drop_duplicates().set_index('chip-version')
        except ValueError as e:
            raise ValueError(
                f"Meta DataFrame must have identical parameters for the same 'chip-version', "
                f"but it appears to have inconsistent duplicates. Error: {e.args[0]}"
                )
        
        pre_array_params.rename(columns=array_params_cols_mapper, inplace=True)
        pre_array_params['n'] = pre_array_params['n'].map(shape_to_n)
        pre_array_params = pre_array_params.to_dict(orient='index')
        
        array_params = {}
        keys_keep = ['n', 's', 'w']
        for chip_version in pre_array_params:
            array_params[chip_version] = {k: pre_array_params[chip_version][k] for k in keys_keep}
            arr_shape = (int(pre_array_params[chip_version]['arr-num-rows']), int(pre_array_params[chip_version]['arr-num-cols']))
            array_params[chip_version]['arr_shape'] = arr_shape
        return array_params
    
    def get_chip_meta(meta_rxn):

        chip_meta = meta_rxn[~meta_rxn['chip-num'].duplicated()].reset_index()[list(chip_meta_cols_mapper.keys())].rename(columns=chip_meta_cols_mapper).set_index('num')
        chip_meta.index = chip_meta.index.astype(int)

        offset_match = chip_meta['offset'].apply(convert_offset, match=True)
        if not offset_match.all():
            raise ValueError(f"Invalid layout array offset format at: {chip_meta['offset'][~offset_match]}")

        chip_meta['offset'] = chip_meta['offset'].apply(convert_offset, match=False)
        chip_meta['extent'] = 'auto'

        return chip_meta
    
    if not isinstance(meta_rxn, pd.DataFrame):
        raise TypeError("meta_rxn must be a pandas DataFrame")

    array_params = get_array_params(meta_rxn)
    chip_meta = get_chip_meta(meta_rxn)

    return array_params, chip_meta
