#!/usr/bin/env python3

import os
import pathlib
from complex_feature_formatting import augment_dataset

def get_current_file_path():
    try:
        return __file__
    except NameError:
        print('__file__ not defined, returning output from `pwd`')
        current_path = pwd
        return current_path

def main():
    current_path = pathlib.Path(get_current_file_path())
    if 'ift6758' not in current_path.parts:
        print('Cannot find parent dir `ift6758`')
        return
    tmp_path_parts = current_path.parts
    base_root_index = 0
    for i in range(0, current_path.parts.count('ift6758')):
        base_root_index += tmp_path_parts.index('ift6758')
        base_root_path = pathlib.Path('/'.join(current_path.parts[:base_root_index+1]))
        if (base_root_path / 'dataset').is_dir():
            break
        tmp_path_parts = tmp_path_parts[base_root_index+1:]
        #raise FileNotFoundError('Cannot find root dir')
    #if not base_root_path.is_dir():
    #    print('Cannot find root dir')
    #    return
    data_input_path = (base_root_path / 'ift6758' / 'dataset' / 'unprocessed')
    data_output_path = (base_root_path / 'ift6758' / 'dataset' / 'complex_engineered')

    augment_dataset(data_input_path=data_input_path, data_output_path=data_output_path, years=None)


if __name__ == '__main__':
    main()
