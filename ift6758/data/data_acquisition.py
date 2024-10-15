# TODO
# Reading into the API and automatically generate the game_ids
# _add_ functionality to be able to aggregate data across multiple seasons

import os
import requests
import time
import json
import threading

class NHLDataFetcher:
    def __init__(self, base_url, save_dir=None):
        # Use environment variable for file path if provided
        self.save_dir = save_dir or os.getenv('DATA_PATH', '../dataset/unprocessed/')
        self.base_url = base_url

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
            with open(filename, 'r', encoding='utf-8') as file:
                #data=file.read()
                #with open(local_file, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    return data, False
                except:
                    pass
        # Otherwise download the data
        url = self.base_url.format(game_id)
        try:
            response = requests.get(url)
            response.raise_for_status()
            response.encoding = 'utf-8'
            data = response.json()
            if 'error' in data:
                print(f"Error found in response for game ID {game_id}: {data['error']}")
                return None, True

            os.makedirs(season_folder, exist_ok=True)  # Ensure directory exists
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(response.text)

            print(f"Successfully downloaded and cached data for game ID: {game_id}")            
            return data, False

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred for game ID {game_id}: {http_err}")
            return None, True
        except Exception as err:
            print(f"An error occurred for game ID {game_id}: {err}")
            return None, True

    def get_game_data_by_id(self, game_id):
        season = game_id[5:9]
        file_path = self.save_dir+'/'+season+'/'+game_id+'.json'
        print(file_path)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    game_data = json.load(f)
                    print('game_data_id',game_data)
            else:
                response = requests.get(self.BASE_URL.format(game_id))
                if response.status_code == 200:
                    game_data = response.json()
                    with open(file_path, 'w') as f:
                        json.dump(game_data, f)
        except:
            game_data = None
        return game_data
    
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
        
        game_types = [2,3]
        game_list = []
        for game_type in game_types:
            for game_number in range(1, num_games+1):
                game_id = f"{year}{game_type:02d}{game_number:04d}"
                data, error = self.get_game_data(game_id)
                game_list.append(data)

        # # Preseason games    
        # for game_number in range(1, num_games+1):
        #     game_id = f"{year}01{game_number:04d}"
        #     data, error = self.get_game_data(game_id)        #     

        #     if error:
        #         print(f"Stopping download of Preseason Games due to an error with game ID: {game_id}")
        #         break
        #     time.sleep(1)        


        # # Regular season games
        # for game_number in range(1, num_games+1):
        #     game_id = f"{year}02{game_number:04d}"
        #     data, error = self.get_game_data(game_id)        #              
      
        #     if error:
        #         print(f"Stopping download of Regular Season Games due to an error with game ID: {game_id}")
        #         break
        #     time.sleep(1)

        # # Playoff games     
        # for round_number in range(1, 5):  # Playoffs have 4 rounds
        #     for matchup_number in range(1, 9):  # Each round has up to 8 matchups
        #         for game_number in range(1, 8):  # Each matchup has up to 7 games
        #             game_id = f"{year}03{round_number}{matchup_number}{game_number}"
        #             data, error = self.get_game_data(game_id)       
        #             if error:
        #                 print(f"Stopping download of Playoff Games due to an error with game ID: {game_id}")
        #                 break
        #             time.sleep(1)
      

        # # All-star game        
        # data, error = self.get_game_data(f"{year}040001")     
        # if error:
        #     print(f"Stopping download of All-Star Game due to an error with game ID: {game_id}")

        print(f"Completed downloading data for the {year}-{int(year)+1} NHL season.")
        


if __name__ == '__main__':
    # Example usage
    base_url = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
    fetcher = NHLDataFetcher(base_url)

#   game_ids = ['2022030411']

#   # Fetch data for each game
#   for game_id in game_ids:
#       data = fetcher.get_game_data(game_id)
#       time.sleep(1)  # Delay to avoid hitting the API too hard

#   year = "2022"
#   # Fetch data for a full season
#   fetcher.get_season_data(year)
    threads = []
    for year in range(2015,2024):
        thread = threading.Thread(target=fetcher.get_season_data, args = (year,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


