#!/usr/bin/env python3

import data_formatting
import pathlib

unprocessed_dir = pathlib.Path('../dataset/unprocessed')
processed_dir = pathlib.Path('../dataset/processed')

def main():
    for file in unprocessed_dir.glob('*/*'):
        data_formatting.process_and_save_json_file(file, processed_dir)

if __name__ == '__main__':
    main()
