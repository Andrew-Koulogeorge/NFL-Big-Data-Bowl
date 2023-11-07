import pandas as pd
import numpy as np
from datetime import datetime
from math import floor

def preprocess_plays_df(plays_df, games_df):
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
def preprocess_players_df(players_df):
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

    # Step 2: Filter variables (EXCLUDES AGE)
    # vars = ['nflId', 'heightInches', 'weight', 'age']
    vars = ['nflId', 'heightInches', 'weight']
    filtered_df = players_df[vars]

    return filtered_df

def preprocess_tracking_df(plays_df_clean, games_df_clean, players_df_clean, tracking_df):
    # Helper function to filter for run plays
    def drop_non_run_plays(run_play_ids, tracking_df):
        print("original tracking df shape: " + str(tracking_df.shape))
        
        # Merge to filter unique combinations from tracking_df
        tracking_df = pd.merge(tracking_df, run_play_ids, on=['gameId', 'playId'], how='inner')

        print("unique play and game id combos: " + str(run_play_ids.shape))
        print("filtered df shape: " + str(tracking_df.shape))
        print("number of merge errors: " + str(len(tracking_df[~tracking_df.set_index(['gameId', 'playId']).index.isin(run_play_ids.set_index(['gameId', 'playId']).index)])))

        return tracking_df
    
    # Helper methods to link dataframes
    def join_play_tracking_data(play_df, tracking_df):
        merged_df = pd.merge(tracking_df, play_df, on=['playId', 'gameId'], how='left')
        print("joined plays and tracking dataframes")
        print("original tracking shape: " + str(tracking_df.shape))
        print("merged data shape: " + str(merged_df.shape))
        print("-------")
        return merged_df 

    def join_player_tracking_data(player_df, tracking_df):
        merged_df = pd.merge(tracking_df, player_df, on=['nflId'], how='left')
        print("joined players and tracking dataframes")
        print("original tracking shape: " + str(tracking_df.shape))
        print("merged data shape: " + str(merged_df.shape))
        print("-------")
        return merged_df

    def join_games_tracking_data(games_df, tracking_df):
        merged_df = pd.merge(tracking_df, games_df, on=['gameId'], how='left')
        print("joined games and tracking dataframes")
        print("original tracking shape: " + str(tracking_df.shape))
        print("merged data shape: " + str(merged_df.shape))
        print("-------")
        return merged_df

    # Helper function to make all plays move in the same direction (right)
    def standardize_direction(merged_df):
        # Home team boolean (1 = home, 0 = away)
        merged_df['isHomeTeam'] = (merged_df['club'] == merged_df['homeTeamAbbr']).astype(int)

        # Offensive team boolean (1 = offense, 0 = defense)
        merged_df['isOnOffense'] = (merged_df['possessionTeam'] == merged_df['club']).astype(int)

        # Play direction
        merged_df['isDirectionLeft'] = (merged_df['playDirection'] == 'left').astype(int)
        
        # Standardize location so all moving towards right end zone
        merged_df['X_std'] = np.where(merged_df['isDirectionLeft'] == 1, 120 - merged_df['x'], merged_df['x'])
        
        merged_df['Y_std'] = np.where(merged_df['isDirectionLeft'] == 1, 160/3  - merged_df['y'], merged_df['y'])

        # Standardize velocity angle
        merged_df['Dir_std'] = np.where(merged_df['isDirectionLeft'] == 1, np.mod(180 + merged_df['dir'], 360), merged_df['dir'])

        # Standardize velocity angle
        merged_df['O_std'] = np.where(merged_df['isDirectionLeft'] == 1, np.mod(180 + merged_df['o'], 360), merged_df['o'])

        # REVIEW THIS!!!!! Set direction and velocity angle of football to be same as the ball carrier
        merged_df.loc[(merged_df['club'] == 'football'),'Dir_std'] = 90
        merged_df.loc[(merged_df['club'] == 'football'),'O_std'] = 90

        return merged_df

    # STEP 0: FILTER RUN PLAYS
    run_play_ids = plays_df_clean[['gameId','playId']].drop_duplicates()
    filtered_df = drop_non_run_plays(run_play_ids, tracking_df)

    # STEP 1: JOIN DATAFRAMES
    merged_df = join_play_tracking_data(plays_df_clean, filtered_df)
    merged_df = join_player_tracking_data(players_df_clean, merged_df)
    merged_df = join_games_tracking_data(games_df_clean, merged_df)

    # STEP 2: STANDARDIZE DIRECTION
    standardized_df = standardize_direction(merged_df)

    # STEP 3: ONE HOT ENCODING
    qualitative_vars = ['club']
    standardized_ohe_df = pd.get_dummies(data = standardized_df, columns= qualitative_vars)
    print("Old df shape:" + str(standardized_df.shape))
    print("New df shape:" + str(standardized_ohe_df.shape))

    # STEP 4: DROP IRRELEVANT FEATURES
    # Drop irrelevant columns
    irrelevent_vars = ['jerseyNumber', 'displayName', 'possessionTeam', 
                    'defensiveTeam', 'playDirection', 'homeTeamAbbr',
                    'x', 'y', 'dir', 'o']
    clean_df = standardized_ohe_df.drop(irrelevent_vars, axis = 1)
    
    return clean_df

# Preprocesses all data
def preprocess_all_df(plays_df, games_df, players_df, tracking_df):
    # Clean plays_df
    print("cleaning plays_df")
    plays_df_clean = preprocess_plays_df(plays_df, games_df)
    print("-----\n")

    # Clean games_df
    print("cleaning games_df")
    games_df_clean = preprocess_games_df(games_df)
    print("-----\n")

    # Clean players_df
    print("cleaning players_df")
    players_df_clean = preprocess_players_df(players_df)
    print("-----\n")

    # Clean tracking_df
    print("cleaning tracking_df")
    clean_df = preprocess_tracking_df(plays_df_clean, games_df_clean, players_df_clean, tracking_df)
    print("-----\n")

    return clean_df

# Method to preprocess plays df ONLY for naive models
def preprocess_plays_df_naive_models(plays_df, games_df, include_nfl_features = False, bin_ouput = False):
    plays_df_in_progress = preprocess_plays_df(plays_df, games_df)

    # One hot encode possession and defense teams
    qualitative_vars = ['defensiveTeam', 'possessionTeam']
    plays_df_ohe = pd.get_dummies(data = plays_df_in_progress, columns= qualitative_vars)

    # Include nfl features if desired
    if include_nfl_features:
        plays_df_full_features = pd.merge(plays_df_ohe, 
                                          plays_df[['gameId','playId',
                                                    'preSnapHomeTeamWinProbability',
                                                    'preSnapVisitorTeamWinProbability',
                                                    'expectedPoints']], 
                                           on=['playId', 'gameId'], how='left')
    else: 
        plays_df_full_features = plays_df_ohe

    # Bin outcome variable
        # Bins:
        # Less than -2
        # -2 - 0 
        # 0 - 1
        # 1 - 2.5
        # 2.5 - 5
        # 5 - 10
        # Greater than 10
    plays_df_clean = plays_df_full_features
    if bin_ouput:
        bins = [float('-inf'), -2, 0, 1, 2.5, 5, 10, float('inf')]
        plays_df_clean['TARGET'] = pd.cut(plays_df_full_features['TARGET'], bins = bins, labels = range(len(bins) - 1))
   

    # Drop NA values
    plays_df_clean = plays_df_clean.dropna()

    return plays_df_clean
