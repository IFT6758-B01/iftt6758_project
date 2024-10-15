# TODO
# Still need to include calculations for when the goal is at even strength, shorthanded, or power play
# Automate converting an entire season to CSV files

import os
import json
import pandas as pd

def parse_game_events(game_data):
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
            period = event.get('periodDescriptor', None)
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
                'empty_net': empty_net
            })

    # Convert the list of events into a Pandas DataFrame
    df = pd.DataFrame(event_data)

    return df

# Open the JSON file, process it, and save the dataframe as a CSV
def process_and_save_json_file(json_filename, csv_save_dir):
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


# Example Usage
game_id = "2022030411"
season = "2022"
season_folder = os.path.join(os.getenv('DATA_PATH', '../dataset/unprocessed/'), season)
filename = os.path.join(season_folder, f'game_{game_id}.json')
process_and_save_json_file(filename, os.path.join(os.getenv('DATA_PATH', '../dataset/processed/'), season))
