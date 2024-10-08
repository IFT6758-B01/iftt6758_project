# TODO
# Reading into the API and automatically generate the game_ids
# _add_ functionality to be able to aggregate data across multiple seasons

import os
import requests
import time

class NHLData:
    def __init__(self, save_dir=None):
        # Use environment variable for file path if provided
        self.save_dir = save_dir or os.getenv('DATA_PATH', '../dataset/unprocessed/')   
        self.preseason_data_list = []
        self.regular_data_list = []
        self.playoff_data_list = []
        self.allstar_data_list = []

    def load_game_data(self, game_id):
        """
        Downloads a single game for the given game_id.
        """
       
        # Create file path
        season = game_id[:4]
        season_folder = os.path.join(self.save_dir, season)
        filename = os.path.join(season_folder, f'game_{game_id}.json')

        # load cached data
        if os.path.exists(filename):
            print(f"Loading cached data for game ID: {game_id}")
            with open(filename, 'r') as file:
                return file.read(), False        

            print(f"Successfully load cached data for game ID: {game_id}")
           
            return data, False


    def load_season_data(self, year):
        """
        load data of a given NHL season (preseason, regular season, playoffs, and all-star).
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
            self.preseason_data_list[year]=data         
            if error:
                print(f"load Preseason Game error, game ID: {game_id}")
                break
            time.sleep(1)

        # Regular season games
        for game_number in range(1, num_games+1):
            game_id = f"{year}02{game_number:04d}"
            data, error = self.get_game_data(game_id)
            self.regular_data_list[year]=data
            if error:
                print(f"loading Regular Season Games error, game ID: {game_id}")
                break
            time.sleep(1)

        # Playoff games
        for round_number in range(1, 5):  # Playoffs have 4 rounds
            for matchup_number in range(1, 9):  # Each round has up to 8 matchups
                for game_number in range(1, 8):  # Each matchup has up to 7 games
                    game_id = f"{year}03{round_number}{matchup_number}{game_number}"
                    data, error = self.get_game_data(game_id)
                    self.playoff_data_list[year]=data
                    if error:
                        print(f"loading Playoff Games data error ,game ID: {game_id}")
                        break
                    time.sleep(1)

        # All-star game
        data, error = self.get_game_data(f"{year}040001")
        self.allstar_data_list[year]=data
        if error:
            print(f"load All-Star Game data error, game ID: {game_id}")

        print(f"Completed load data for the {year}-{int(year)+1} NHL season.")


