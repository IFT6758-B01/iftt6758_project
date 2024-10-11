# TODO
# Still need to include calculations for when the goal is at even strength, shorthanded, or power play
# Automate converting an entire season to CSV files

import os
import json
import pathlib
import sys
import re
import pandas as pd
from typing import Tuple

def parse_game_events(game_data: dict) -> pd.DataFrame:
    """
    Parses the JSON response of a game to extract 'shot-on-goal' and 'goal' events and converts them into a Pandas DataFrame.

    Parameters:
        game_data (dict): JSON response of a single game's events.

    Returns:
        pd.DataFrame: A dataframe containing the filtered and formatted events data.
    """
    events = game_data.get('plays', [])
    game_id = game_data.get('id', '')

    # List to store parsed events
    event_data = []

    # Loop through each event in the game data
    for event in events:
        # Filter for shot-on-goal or goal
        event_type = event.get('typeDescKey', '')
        if event_type in ['shot-on-goal', 'goal']:
            # Extract time, period, and event fields
            time_in_period = event.get('timeInPeriod', None)
            time_remaining = event.get('timeRemaining', None)
            period = event.get('periodDescriptor', None).get('number', None)
            event_id = event.get('eventId', None)

            # Team that took the shot
            team_id = event.get('details', {}).get('eventOwnerTeamId', None)

            # Coordinates of the shot
            x_coord = event.get('details', {}).get('xCoord', None)
            y_coord = event.get('details', {}).get('yCoord', None)

            # Shooter and goalie info
            shooter_id = event.get('details', {}).get('shootingPlayerId', None)
            scoring_player_id = event.get('details', {}).get('scoringPlayerId', None)
            goalie_id = event.get('details', {}).get('goalieInNetId', None)

            # Shot type
            shot_type = event.get('details', {}).get('shotType', None)

            # Zone Code for figuring out which side the team is on
            zone_code = event.get('details', {}).get('zoneCode', None)


            # Check if it was on an empty net (no goalie present)
            empty_net = goalie_id is None

            # Append the event data to the list
            event_data.append({
                'game_id': game_id,
                'event_id': event_id,
                'event_type': event_type,
                'period': period,
                'time_in_period': time_in_period,
                'time_remaining': time_remaining,
                'team_id': team_id,
                'x_coord': x_coord,
                'y_coord': y_coord,
                'shooter_id': shooter_id or scoring_player_id,  # Use scoring player ID if it's a goal
                'goalie_id': goalie_id,
                'shot_type': shot_type,
                'empty_net': empty_net,
                'zone_code': zone_code
            })

    # Convert the list of events into a Pandas DataFrame
    df = pd.DataFrame(event_data)

    return df

# Backup of original process_and_save_json_file --> added `_` in front
# Open the JSON file, process it, and save the dataframe as a CSV
def _process_and_save_json_file(json_filename, csv_save_dir):
    # Open the file
    with open(json_filename, 'r') as file:
        game_data = json.load(file)  # Load the data into a dictionary

    # Process the data using the parse_game_events function
    df = parse_game_events(game_data)

    # Get the game ID from the data for naming the file
    game_id = game_data.get('id', None)

    # Create the directory if it doesn't exist
    os.makedirs(csv_save_dir, exist_ok=True)

    # CSV file path
    csv_filename = os.path.join(csv_save_dir, f'game_{game_id}.csv')

    # Save the dataframe to CSV
    df.to_csv(csv_filename, index=False)

    print(f"Data for game {game_id} has been saved to {csv_filename}.")


def gather_and_check_paths(DATA_INPUT_PATH: str = None, DATA_OUTPUT_PATH: str = None) -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Gather DATA_INPUT_PATH and DATA_OUTPUT_PATH in the following order:
      Argument fed in the function, else
      System Environment Varaible as 'NHL_DATA_[INPUT|OUTPUT]_PATH', else
      Default directory '{ROOT_DIR}/dataset/[unprocessed|processed]' with ROOT_DIR as root directory of currently executed script

      Parameters:
          DATA_INPUT_PATH·(str) : Input directory of unprocessed data
          DATA_OUTPUT_PATH·(str) : Output directory of processed (DataFrames) data
      Returns:
          DATA_INPUT_PATH (pathlib.Path)
          DATA_OUTPUT_PATH (pathlib.Path)
    """

    #Get absolute path of currently run script
    ##From https://stackoverflow.com/a/595317
    EXEC_PY_PATH = pathlib.Path(sys.argv[0]).absolute()
    #Setting dir 'ift6758' as ROOT_DIR
    ROOT_DIR = EXEC_PY_PATH.parents[1]
    #Check if DATA_INPUT_PATH and DATA_OUTPUT_PATH have been set and get their absolute paths
    DATA_INPUT_PATH = DATA_INPUT_PATH or os.getenv('NHL_DATA_INPUT_PATH', f'{ROOT_DIR}/dataset/unprocessed/')
    DATA_INPUT_PATH = pathlib.Path(DATA_INPUT_PATH).absolute()
    DATA_OUTPUT_PATH = DATA_OUTPUT_PATH or os.getenv('NHL_DATA_OUTPUT_PATH', f'{ROOT_DIR}/dataset/processed/')
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
    return DATA_INPUT_PATH, DATA_OUTPUT_PATH

def process_and_save_json_file(DATA_INPUT_PATH : pathlib.Path, DATA_OUTPUT_PATH : pathlib.Path) -> None:
    """
    Process all .json files found in DATA_INPUT_PATH, convert to Pandas DataFrame and save them to csv in DATA_OUTPUT_PATH

    Parameters:
        DATA_INPUT_PATH (pathlib.Path) : Input directory of unprocessed data
                                         Assumes the following hierarchy : DATA_INPUT_PATH/{season_folder}/{gameid}*.json
        DATA_OUTPUT_PATH (pathlib.Path) : Output directory of processed (DataFrames) data
                                          Copies hierarchy of DATA_INPUT_PATH for naming output csvs

    Returns:
        None
    """

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
                df_game = parse_game_events(game_dict)
                df_game.to_csv(output_file)
                print(f'Saved csv of dataframe to {output_file}')



def main():
  DATA_INPUT_PATH, DATA_OUTPUT_PATH = gather_and_check_paths()
  process_and_save_json_file(DATA_INPUT_PATH, DATA_OUTPUT_PATH)

if __name__ == '__main__' :
    main()


'''
    # Example Usage
    game_id = "2022030411"
    season = "2022"
    season_folder = os.path.join(os.getenv('DATA_PATH', '../dataset/unprocessed/'), season)
    filename = os.path.join(season_folder, f'game_{game_id}.json')
    process_and_save_json_file(filename, os.path.join(os.getenv('DATA_PATH', '../dataset/processed/'), season))
'''
