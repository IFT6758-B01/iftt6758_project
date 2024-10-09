# TODO
# Reading into the API and automatically generate the game_ids
# _add_ functionality to be able to aggregate data across multiple seasons

import os
import requests
import time
from sys import argv

class NHLDataFetcher:
    def __init__(self, base_url, save_dir=None):
        # Use environment variable for file path if provided
        self.get_root_path = self._get_root_path()
        self.save_dir = save_dir or os.getenv('DATA_PATH', f'{self.get_root_path}/dataset/unprocessed/')
        self.base_url = base_url

    def _get_exec_path(self):
        return os.path.abspath(argv[0]).split('/')[:-1]

    def _get_root_path(self):
        self.exec_path = self._get_exec_path()
        return '/'.join(self.exec_path[:-1])

    def get_game_data(self, game_id):
        """
        Downloads a single game for the given game_id.
        """
        # Create file path
        season = game_id[:4]
        season_folder = os.path.join(self.save_dir, season)
        filename = os.path.join(season_folder, f'game_{game_id}.json')


        # Check if the data is already cached
        if os.path.exists(filename):
            print(f"Loading cached data for game ID: {game_id}")
            with open(filename, 'r') as file:
                return file.read(), False

        # Otherwise download the data
        url = self.base_url.format(game_id)
        try:
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            if 'error' in data:
                print(f"Error found in response for game ID {game_id}: {data['error']}")
                return None, True

            os.makedirs(season_folder, exist_ok=True)  # Ensure directory exists
            with open(filename, 'w') as file:
                file.write(response.text)

            print(f"Successfully downloaded and cached data for game ID: {game_id}")
            return data, False

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred for game ID {game_id}: {http_err}")
            return None, True
        except Exception as err:
            print(f"An error occurred for game ID {game_id}: {err}")
            return None, True

    def get_season_data(self, year):
        """
        Downloads all games for a given NHL season (preseason, regular season, playoffs, and all-star).
        """
        if int(year) >= 2021:
            num_games = 1353
        elif int(year) >= 2017:
            num_games = 1271
        else:
            num_games = 1230

        # Preseason games
        for game_number in range(1, num_games+1):
            game_id = f"{year}01{game_number:04d}"
            data, error = self.get_game_data(game_id)
            if error:
                print(f"Stopping download of Preseason Games due to an error with game ID: {game_id}")
                break
            time.sleep(1)

        # Regular season games
        for game_number in range(1, num_games+1):
            game_id = f"{year}02{game_number:04d}"
            data, error = self.get_game_data(game_id)
            if error:
                print(f"Stopping download of Regular Season Games due to an error with game ID: {game_id}")
                break
            time.sleep(1)

        # Playoff games
        for round_number in range(1, 5):  # Playoffs have 4 rounds
            for matchup_number in range(1, 9):  # Each round has up to 8 matchups
                for game_number in range(1, 8):  # Each matchup has up to 7 games
                    game_id = f"{year}03{round_number}{matchup_number}{game_number}"
                    data, error = self.get_game_data(game_id)
                    if error:
                        print(f"Stopping download of Playoff Games due to an error with game ID: {game_id}")
                        break
                    time.sleep(1)

        # All-star game
        data, error = self.get_game_data(f"{year}040001")
        if error:
            print(f"Stopping download of All-Star Game due to an error with game ID: {game_id}")

        print(f"Completed downloading data for the {year}-{int(year)+1} NHL season.")



if __name__ == '__main__':
  # Example usage
  base_url = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
  fetcher = NHLDataFetcher(base_url)

  game_ids = ['2022030411']

  # Fetch data for each game
  for game_id in game_ids:
      data = fetcher.get_game_data(game_id)
      time.sleep(1)  # Delay to avoid hitting the API too hard

  year = "2022"
  # Fetch data for a full season
  fetcher.get_season_data(year)
