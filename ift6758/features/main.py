#!/usr/bin/env python3

'''
TODO available below
'''
import data_formatting
import pathlib
import os
import sys
import pandas
import json
import re

#Get absolute path of currently run script
##From https://stackoverflow.com/a/595317
EXEC_PY_PATH = pathlib.Path(sys.argv[0]).absolute()
#Setting dir 'ift6758' as ROOT_DIR
ROOT_DIR = EXEC_PY_PATH.parents[1]
#Check if DATA_INPUT_PATH and DATA_OUTPUT_PATH have been set and get their absolute paths
DATA_INPUT_PATH = os.getenv('NHL_DATA_INPUT_PATH', f'{ROOT_DIR}/dataset/unprocessed/')
DATA_INPUT_PATH = pathlib.Path(DATA_INPUT_PATH).absolute()
DATA_OUTPUT_PATH = os.getenv('NHL_DATA_OUTPUT_PATH', f'{ROOT_DIR}/dataset/processed/')
DATA_OUTPUT_PATH = pathlib.Path(DATA_OUTPUT_PATH).absolute()
#Check that those paths exist
##DATA_INPUT_PATH must exists, else raise FileNotFoundError
##DATA_OUTPUT_PATH could not exist, create it
if not DATA_INPUT_PATH.exists():
    raise FileNotFoundError(f'{DATA_INPUT_PATH} does not exists, cannot process data')
if not DATA_OUTPUT_PATH.exists():
    print(f'Could not find output directory {DATA_OUTPUT_PATH}')
    print('Creating it..')
    os.makedirs(DATA_OUTPUT_PATH)

def main():
    for game_json_file in DATA_INPUT_PATH.rglob("*.json"):
        game_title = game_json_file.parts[-1]
        game_title_csv = re.sub('json$', 'csv', game_title)
        season_folder = game_json_file.parts[-2]
        output_file = DATA_OUTPUT_PATH.joinpath(season_folder, game_title_csv)
        #Check if processed file already exists
        if output_file.exists():
            print(f'File {output_file} already exists. Skipping')
        else:
            #Check if DATA_OUTPUT_PATH/season_folder exists, else create it
            if not output_file.parent.exists():
                os.mkdir(output_file.parent)
            with open(game_json_file, 'r') as open_file:
                print(f'Processing {game_json_file}..')
                game_dict = json.load(open_file)
                df_game = data_formatting.parse_game_events(game_dict)
                df_game.to_csv(output_file)
                print(f'Saved csv of dataframe to {output_file}')

if __name__ == '__main__':
    main()


'''
Steps of main


    #For every game JSON in DATA_INPUT_PATH (if specified, else {ROOT_DIR}/data/dataset/unprocessed/{season})
      #DATA_INPUT_PATH must be at unprocessed/ level
      #If using os.listdir, we want filenames to be './{season}/{gameid}.json'
    #Check if processed version ({ROOT_DIR}/data/dataset/processed/{game}) exists
      #If exists, either wait for user input for overwrite/skip or read possible flags provided when called main
    #Read game into dict via json.load()
    #Check if actually JSON/dict with required fields (id, plays) for processing
    #Once all checks are done, pass dict to data_formatting.parse_game_events() -> pd.DataFrame
    #Save df to csv into DATA_OUTPUT_PATH (if specified, else {ROOT_DIR}/data/dataset/processed/{season})
      #DATA_OUTPUT_PATH must be at unprocessed/ level
'''



'''
TODO

= Assert that necessary keys ('id', 'plays') are in json_dict before sending to data_formatting.parse_game_events
    #assert 'id' in game_dict.keys(), f'Missing `id` key in {game_json_file}'
    #assert 'plays' in game_dict.keys(), f'Missing `plays` key in {game_json_file}'
= ArgParse with arg '-y' to be able to implement user input() on if creating DATA_OUTPUT_DIR if not existant?
    #input_create_outdir = input('Create it ? [y/n]') if args.yesall is not None else 'y'
    #if input_create_outdir == 'y':
    #    os.makedirs(DATA_OUTPUT_PATH)
= Encapsulate login in main() into another function to keep main() clean
'''
