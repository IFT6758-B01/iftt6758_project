import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import re
import os
import json
import requests

os.sys.path.append((pathlib.Path(__file__) / '..' ).resolve().absolute())
#from serving import app




st.title("HOCKEY VIZ TOOL")

URL = 'http://0.0.0.0:8080'
URL_TEAMNAME = "https://api.nhle.com/stats/rest/en/team"
URL_GAMESTORY = "https://api-web.nhle.com/v1/wsc/game-story/"

with st.sidebar:
    # Add input for the sidebar    
    st.sidebar.header("Select model")
    model_selection_form = st.sidebar.form('Model')
    #model_selection_form.write('Default entity: IFT6758_2024-B01')
    #workspace = model_selection_form.text_input('Workspace:', placeholder='ms2-logistic-regression')
    workspace = model_selection_form.text_input('Workspace:', value='IFT6758_2024-B01')
    model = model_selection_form.text_input('Model:', value='Distance_Angle')
    model_version = model_selection_form.text_input('Version:', value='V0')
    model_submit = model_selection_form.form_submit_button('Submit')
    if model_submit and (model == '' or model_version == '' or workspace == ''):
        model_selection_form.warning('Empty field Workspace/Model/Version')
    if model_submit and (model != '' and model_version != '' and workspace != ''):
        print(type(model))
        print(type(model_version))
        st.sidebar.write('Fetching model', model, 'with version', model_version, 'in', workspace)
        model_selection_json = {
            'workspace': workspace,
            'model': model,
            'version': model_version
        }
        response = requests.post(f"{URL}/download_registry_model", json = model_selection_json)
        st.sidebar.write(response.status_code, response.reason)
        if response.status_code != 200:      
            st.sidebar.write(response.content)



#list_downloaded_games = pd.read_csv(pathlib.Path('./ift6758/dataset/complex_engineered/augmented_data.csv'))['game_id'].unique().tolist()
@st.cache_data(max_entries=1)
def list_downloadable_games():
    gen_possible_games = [possible_gid for possible_gid in range(2015020001, 2024029999) if re.match('20[1-2][0-9]02[0-1][0-9]{3}', str(possible_gid))]
    list_downloadable_games = [gid for gid in gen_possible_games if int(str(gid)[-4:]) < 1350]
    return list_downloadable_games
#list_available_games = list_downloaded_games + list_downloadable_games


with st.container():
    # Add Game ID input  
    game_id_container = st.container()
    list_downloadable_games = list_downloadable_games()
    game_id_selectbox = game_id_container.selectbox('Game ID', list_downloadable_games)
    game_id_submit = game_id_container.button('Ping game')
    if game_id_submit:
        request = { 'game_id': game_id_selectbox }
        response = requests.post(URL+'/process_game', json=request)



with st.container():
    #Add Game info,predictions,data   
    predictions_container = st.container()
    if not game_id_submit:
        st.write('Waiting for game input')
    else:
        response_json = response.json()
        if response_json == '':
            predictions_container.error('Got empty response, is game valid ?')      
        # Handle response based on content type
        if isinstance(response_json, dict):
            # If it's a message, check for "No new events" case
            message = response_json.get("message", "")
            st.write(f"No new events to process")  
            # if message == "No new events to process.":
            #     st.write(f"No new events to process")                 
            # else:
            #     st.write(f"Unexpected message received: {message}")                    
        else:            
            if not str(response.status_code).startswith('2'):
                with open('./serving/flask.log', 'r') as logs:
                    tail_logs = ''
                    for line in (logs.readlines() [-20:]):
                        tail_logs += line+'\n'
                    st.error('An error occured while generating predictions, please refer to logs\n\n')
                    st.error(tail_logs)
            else:
                if "xg_df" not in st.session_state:
                    #st.write(response_json)
                    df = pd.DataFrame(
                    data=[ (game_event.get('predicted_probabilities'), game_event.get('team_id')) for game_event in response_json ],
                    columns=['goal_proba', 'team_id']
                    )
                    df_all= pd.json_normalize(response_json) 

                    home_game_xG = df.groupby(by='team_id').sum().reset_index()  

                    # get all team names list  
                    try:
                        response = requests.get(URL_TEAMNAME )
                        response.raise_for_status()
                        data = response.json()            
                        df_team = pd.json_normalize(data['data'])
                    except requests.Timeout:
                        st.error(f"Request to {URL_TEAMNAME } timed out.") 
                    except ConnectionError:
                        st.error(f"Error connecting to {URL_TEAMNAME }. Check your internet connection.")  
                    except requests.HTTPError as http_err:
                        st.error(f"HTTP error occurred: {http_err}")
                        if response.status_code == 404:
                            st.error(f"Resource not found at {URL_TEAMNAME }") #More specific error message
                    except requests.RequestException as req_err:  # Catch other request exceptions
                        st.error(f"A request error occurred: {req_err}")                  
                    except ValueError as json_err: #Catch json decoding errors
                        st.error(f"Decoding JSON failed: {json_err}")
                        st.error(f"Response text: {response.text}") #Print response text to debug                 
                    except Exception as e: #Catch other errors
                        st.error(f"An unexpected error occurred: {e}")

                    # add one column 'name'  
                    home_game_xG['name'] = home_game_xG['team_id'].map(df_team.set_index('id')['fullName'])
                    

                    # print team name
                    team_1 = home_game_xG.at[0, 'name']
                    team_2 = home_game_xG.at[1, 'name']                
                    st.write(f"Game: {game_id_selectbox}")
                    st.header(f"{team_1} VS {team_2}")
                    
                    st.session_state.xg_df = home_game_xG
                    #away_game_xG·=·

                    # Get period and left time from last row as current period          
                    last_period = df_all.iloc[-1]['period'] # -1 for last row
                    last_time_remaining = df_all.iloc[-1]['time_remaining']
                    st.write(f"Period: {last_period}, Time: {last_time_remaining} - left")            

                    #print Score ( predict Score)      
                    #get socre info 
                    url_gamestory = f"{URL_GAMESTORY}{game_id_selectbox}"     
                    try:     
                        response = requests.get(url_gamestory)
                        response.raise_for_status()
                        data = response.json() 
                        score_1 = round(data['awayTeam']['score'])
                        score_2 = round(data['homeTeam']['score'])      
                        predict_score_1 = round(home_game_xG.at[0, 'goal_proba'],1)
                        predict_score_2 = round(home_game_xG.at[1, 'goal_proba'],1)
                        diff_1= round(predict_score_1-score_1,1)
                        diff_2= round(predict_score_2-score_2,1)
                        col1, col2= st.columns(2)
                        with col1:            
                            st.metric(label=f"{team_1} Current(Predict)", value=f"{score_1}({predict_score_1})", delta=f"{diff_1}")
                        with col2:      
                            st.metric(label=f"{team_2}", value=f"{score_2}({predict_score_2})", delta=f"{diff_2}")
                    except requests.Timeout:
                        st.error(f"Request to { url_gamestory } timed out.") 
                    except ConnectionError:
                        st.error(f"Error connecting to { url_gamestory  }. Check your internet connection.")  
                    except requests.HTTPError as http_err:
                        st.error(f"HTTP error occurred: {http_err}")
                        if response.status_code == 404:
                            st.error(f"Resource not found at { url_gamestory }") #More specific error message
                    except requests.RequestException as req_err:  # Catch other request exceptions
                        st.error(f"A request error occurred: {req_err}")                  
                    except ValueError as json_err: #Catch json decoding errors
                        st.error(f"Decoding JSON failed: {json_err}")
                        st.error(f"Response text: {response.text}") #Print response text to debug                 
                    except Exception as e: #Catch other errors
                        st.error(f"An unexpected error occurred: {e}")
                    
                    # show data used for predictions                    
                    st.header(f"Data used for predictions")
                    df_all
                    
                else:                    
                    st.session_state.xg_df
    
