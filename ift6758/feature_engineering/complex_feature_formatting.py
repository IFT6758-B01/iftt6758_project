import os
import json
import pathlib
import re
import pandas as pd
import numpy as np
import simple_features as sf


def calculate_game_seconds(period_time, period_number):
    """
    Convert period time to game seconds, accounting for previous periods.

    Parameters:
        period_time (str): Time in the current period in MM:SS format (e.g., '12:34').
        period_number (int): The period number (e.g., 1, 2, 3).

    Returns:
        int: The total game seconds elapsed at the given period time.
    """
    if pd.isna(period_time) or period_number is None:
        return None

    # Parse minutes and seconds from the period time
    try:
        minutes, seconds = map(int, period_time.split(':'))
    except ValueError:
        return None  # Return None if period_time is invalid

    # Convert period time to seconds
    period_seconds = minutes * 60 + seconds

    # Account for time elapsed in previous periods (20 minutes per period)
    previous_period_seconds = (period_number - 1) * 20 * 60

    # Total game seconds
    game_seconds = period_seconds + previous_period_seconds

    return game_seconds


def parse_game_events(game_data: dict) -> pd.DataFrame:
    """
    Parses the JSON response of a game to extract 'shot-on-goal' and 'goal' events, along with
    information from the immediately preceding event.

    Parameters:
        game_data (dict): JSON response of a single game's events.

    Returns:
        pd.DataFrame: A dataframe containing the events data with additional features.
    """
    events = game_data.get('plays', [])
    game_id = game_data.get('id', '')

    # List to store parsed events
    event_data = []
    previous_event = None

    # Loop through each event in the game data
    for event in events:
        event_type = event.get('typeDescKey', '')
        if event_type in ['shot-on-goal', 'goal']:
            # Extract details of the current event
            period = event.get('periodDescriptor', {}).get('number', None)
            time_in_period = event.get('timeInPeriod', None)
            time_remaining = event.get('timeRemaining', None)
            event_id = event.get('eventId', None)
            team_id = event.get('details', {}).get('eventOwnerTeamId', None)
            x_coord = event.get('details', {}).get('xCoord', None)
            y_coord = event.get('details', {}).get('yCoord', None)
            shooter_id = event.get('details', {}).get('shootingPlayerId', None)
            scoring_player_id = event.get('details', {}).get('scoringPlayerId', None)
            goalie_id = event.get('details', {}).get('goalieInNetId', None)
            shot_type = event.get('details', {}).get('shotType', None)
            zone_code = event.get('details', {}).get('zoneCode', None)
            empty_net = goalie_id is None

            game_seconds = calculate_game_seconds(time_in_period, period)

            # Initialize previous event details
            last_event_type = None
            last_x_coord = None
            last_y_coord = None
            time_from_last_event = None
            distance_from_last_event = None

            # If there's a previous event, calculate new features
            if previous_event:
                last_event_type = previous_event['event_type']
                last_x_coord = previous_event['x_coord']
                last_y_coord = previous_event['y_coord']

                # Time difference from the last event
                if game_seconds is not None and previous_event['game_seconds'] is not None:
                    time_from_last_event = abs(game_seconds - previous_event['game_seconds'])

                # Distance from the last event
                if x_coord is not None and y_coord is not None and last_x_coord is not None and last_y_coord is not None:
                    distance_from_last_event = np.sqrt(
                        (x_coord - last_x_coord) ** 2 + (y_coord - last_y_coord) ** 2
                    )

            # Additional features
            rebound = last_event_type in ['shot-on-goal', 'goal']  # True if the last event was a shot or goal

            # Append the event data to the list
            event_data.append({
                'game_id': game_id,
                'event_id': event_id,
                'event_type': event_type,
                'period': period,
                'time_in_period': time_in_period,
                'time_remaining': time_remaining,
                'game_seconds': game_seconds,
                'team_id': team_id,
                'x_coord': x_coord,
                'y_coord': y_coord,
                'shooter_id': shooter_id or scoring_player_id,
                'goalie_id': goalie_id,
                'shot_type': shot_type,
                'empty_net': empty_net,
                'zone_code': zone_code,
                'last_event_type': last_event_type,
                'last_x_coord': last_x_coord,
                'last_y_coord': last_y_coord,
                'time_from_last_event': time_from_last_event,
                'distance_from_last_event': distance_from_last_event,
                'rebound': rebound
            })
        else:
            # if the current event is not a shot or goal, still extract event type and coordinates as well as game seconds
            event_type = event.get('typeDescKey', '')
            x_coord = event.get('details', {}).get('xCoord', None)
            y_coord = event.get('details', {}).get('yCoord', None)
            game_seconds = calculate_game_seconds(event.get('timeInPeriod', None), event.get('periodDescriptor', {}).get('number', None))

        # Update the previous event
        previous_event = {
            'event_type': event_type,
            'x_coord': x_coord,
            'y_coord': y_coord,
            'game_seconds': game_seconds
        }

    # Convert the list of events into a Pandas DataFrame
    df = pd.DataFrame(event_data)
    return df


def augment_data_complex(all_data):
    """
    Augments the raw data with new features and saves the augmented data to a CSV file.

    Parameters:
        input_path (pathlib.Path): Path to the directory containing the raw data CSV files.
        output_path (pathlib.Path): Path to the directory where augmented data will be saved.

    Returns:
        None
    """
    # Segment the data by team, game, and period
    segmented_data = sf.segment_shot_data(all_data)

    # Calculate rebound metrics for each period
    for team_id, games in segmented_data.items():
        for game_id, periods in games.items():
            for period, period_data in periods.items():
                segmented_data[team_id][game_id][period] = calculate_rebound_metrics(period_data)

    # Aggregate the data and calculate new metrics
    df_aggregate = sf.aggregate_data(segmented_data)

    return df_aggregate


def calculate_rebound_metrics(period_data):
    """
    Calculates new metrics for rebound shots in a period and recombines with non-rebound shots.

    Parameters:
        period_data (pd.DataFrame): DataFrame containing shots for a team in a period.

    Returns:
        pd.DataFrame: DataFrame with rebound metrics added, recombined with non-rebound shots.
    """
    # Filter for rebound events
    df_rebounds = period_data[period_data['rebound'] == True].copy()

    if not df_rebounds.empty:
        # Determine the goal location based on the shots in the period
        goal_location = sf.determine_goal_location(period_data)

        # Calculate distance and angle for previous shots (rebound-related metrics)
        last_distances, last_angles = zip(*df_rebounds.apply(
            lambda row: sf.calculate_distance_and_angle(row['last_x_coord'], row['last_y_coord'], net_x=goal_location[0], net_y=goal_location[1]), axis=1
        ))

        # Calculate change in shot angle and speed for rebounds
        df_rebounds['change_in_shot_angle'] = df_rebounds['angle_from_net'] - last_angles
        df_rebounds['speed'] = df_rebounds['distance_from_last_event'] / df_rebounds['time_from_last_event']

        # Fill missing or invalid values in speed with 0 (e.g., division by zero)
        #df_rebounds['speed'].fillna(0, inplace=True)

        # Identify non-rebound shots
        df_non_rebounds = period_data[period_data['rebound'] == False].copy()

        # Combine rebound and non-rebound DataFrames
        combined_data = pd.concat([df_rebounds, df_non_rebounds], ignore_index=True)

        # Ensure the combined data retains the original order
        combined_data.sort_index(inplace=True)

        return combined_data
    else:
        return period_data



def process_and_save_json_file(DATA_INPUT_PATH : pathlib.Path, DATA_OUTPUT_PATH : pathlib.Path, from_timestamp: str = None) -> None:
    """
    Process all .json files found in DATA_INPUT_PATH, convert to Pandas DataFrame and save them to csv in DATA_OUTPUT_PATH

    Parameters:
        DATA_INPUT_PATH (pathlib.Path) : Input directory of unprocessed data
                                         Assumes the following hierarchy : DATA_INPUT_PATH/{season_folder}/{gameid}*.json
        DATA_OUTPUT_PATH (pathlib.Path) : Output directory of processed (DataFrames) data
                                          Copies hierarchy of DATA_INPUT_PATH for naming output csvs
        from_timestamp (str) : Timestamp from epoch in seconds (datetime.datetime.strftime) from which files should be processed

    Returns:
        None
    """
    for game_json_file in DATA_INPUT_PATH.rglob("**/game*.json"):
        #Check if file is older than `from_timestamp`
        if from_timestamp is not None and int(game_json_file.stat().st_ctime) <= int(from_timestamp):
            continue
        game_title = game_json_file.parts[-1]
        game_title_csv = re.sub('json$', 'csv', game_title)
        season_folder = game_json_file.parts[-2]
        output_file = DATA_OUTPUT_PATH.joinpath(season_folder, game_title_csv)
        #Check if processed file already exists
        if output_file.exists():
            print(f'File {output_file} already exists. Skipping')
            continue
        #Check if DATA_OUTPUT_PATH/season_folder exists, else create it
        if not output_file.parent.exists():
            os.mkdir(output_file.parent)
        with open(game_json_file, 'r') as open_file:
            print(f'Processing {game_json_file}..')
            game_dict = json.load(open_file)
            df_game = parse_game_events(game_dict)
            df_game = augment_data_complex(df_game)
            df_game.to_csv(output_file)
            print(f'Saved csv of dataframe to {output_file}')




input_directory = "../dataset/unprocessed/2017"
output_directory = "../dataset/complex_engineered/"

input_path = pathlib.Path(input_directory)
output_path = pathlib.Path(output_directory)
output_path.mkdir(parents=True, exist_ok=True)

process_and_save_json_file(input_path, output_path)

