# TODO
# Reading into the API and automatically generate the game_ids
# _add_ functionality to be able to aggregate data across multiple seasons

import os
import requests
import time

class NHLDataFetcher:
    def __init__(self, base_url, save_dir=None):
        # Use environment variable for file path if provided
        self.save_dir = save_dir or os.getenv('DATA_PATH', '../downloaded_data/')
        self.base_url = base_url
    
    def get_season_data(self, game_id):
        # Create file path
        filename = f'{self.save_dir}game_{game_id}.json'
        
        # Check if the data is already cached
        if os.path.exists(filename):
            print(f"Loading cached data for game ID: {game_id}")
            with open(filename, 'r') as file:
                return file.read()
        
        # Otherwise download the data
        url = self.base_url.format(game_id)
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            os.makedirs(self.save_dir, exist_ok=True)  # Ensure directory exists
            with open(filename, 'w') as file:
                file.write(response.text)
            
            print(f"Successfully downloaded and cached data for game ID: {game_id}")
            return response.text
        
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred for game ID {game_id}: {http_err}")
        except Exception as err:
            print(f"An error occurred for game ID {game_id}: {err}")


# Example usage
base_url = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
fetcher = NHLDataFetcher(base_url)

game_ids = ['2022030411']

# Fetch data for each game
for game_id in game_ids:
    data = fetcher.get_season_data(game_id)
    time.sleep(1)  # Delay to avoid hitting the API too hard