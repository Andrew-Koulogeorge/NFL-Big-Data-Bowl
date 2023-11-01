import pandas as pd
import numpy as np
from datetime import datetime
from math import floor

class Preprocessing:
    def preprocess_play_df(plays_df, games_df):
        # Filter for only run plays
        run_plays = plays_df[-plays_df['playDescription'].str.contains('pass')]
        
        run_plays_reduced = run_plays.drop(['yardlineSide', 'yardlineNumber','yardlineSide', 'passResult', 'passLength',
        'penaltyYards', 'playResult', 'playNullifiedByPenalty', 'passProbability', 
        'preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability', 
        'homeTeamWinProbabilityAdded', 'playDescription',
        'visitorTeamWinProbilityAdded', 'expectedPoints', 'expectedPointsAdded',
        'foulName1', 'foulName2', 'foulNFLId1', 'foulNFLId2'], axis = 1)
        
        # Rename outcome variable
        run_plays_reduced = run_plays_reduced.rename(columns = {'prePenaltyPlayResult':'TARGET'})

        # Rename pre-snap variables
        pre_snap_vars = ['yardsToGo', 'yardlineNumber', 'gameClock']
        for var in pre_snap_vars:
            new_name = "preSnap" + var[0].upper() + var[1:]
        run_plays_reduced = run_plays_reduced.rename(columns = {var:new_name})

        # Convert pre snap game clock into seconds
        run_plays_reduced['preSnapGameClockSec'] = pd.to_timedelta('00:' + run_plays_reduced['preSnapGameClock']).dt.total_seconds().astype(int)
        run_plays_reduced = run_plays_reduced.drop(columns=['preSnapGameClock'], axis = 1)
    
        # One hot encode qualitative variables
        # qualitative_vars = ["possessionTeam", "defensiveTeam", "offenseFormation"]
        qualitative_vars = ["offenseFormation", 'ballCarrierDisplayName']
        run_plays_ohe = pd.get_dummies(data = run_plays_reduced, columns= qualitative_vars)

        # Merge with games data
        run_df_clean = games_df[['gameId', 'week']].merge(run_plays_ohe, on='gameId')

        print("final plays data shape: " + str(run_df_clean.shape))
        
        return run_df_clean  
    
    # Method to pre-process games dataframe
    def preprocess_games_df(games_df):
        # Select key variables
        filtered_df = games_df[['gameId', 'homeTeamAbbr']]
        return filtered_df
    
    # Function that preproceses player dataframe
    def preprocess_players(players_df):
        # Step 0: Convert height to inches
        players_df['heightInches'] = players_df['height'].str.split('-').apply(lambda x: int(x[0]) * 12 + int(x[1]))
        
        # Step 1: Compute age from birthdate
        # Step 1a: Convert 'birthDate' to datetime (if it's not already in datetime format)
        players_df['birthDate'] = pd.to_datetime(players_df['birthDate'], errors='coerce')

        # Step 1b: Calculate age using vectorized operations
        today = datetime.today()
        players_df['age'] = today.year - players_df['birthDate'].dt.year

        # Step 1c: Handle NaN birthdates
        players_df.loc[pd.isnull(players_df['birthDate']), 'age'] = np.NaN

        # Step 2: Filter variables
        vars = ['nflId', 'heightInches', 'weight', 'age']
        filtered_df = players_df[vars]

        return filtered_df
    
    