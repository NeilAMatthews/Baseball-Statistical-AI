import pandas as pd
from pybaseball import statcast, playerid_lookup
import os
from datetime import datetime
import numpy as np

CACHE_FILE = 'statcast_2024_sample.csv'

def fetch_data(start_date='2024-04-01', end_date='2024-07-01'):
    """
    Fetches Statcast data for a given date range.
    Returns a DataFrame containing pitch-by-pitch data.
    """
    if os.path.exists(CACHE_FILE):
        print(f"Loading data from {CACHE_FILE}...")
        return pd.read_csv(CACHE_FILE)
    
    print(f"Fetching Statcast data from {start_date} to {end_date}...")
    # statcast returns a dataframe
    data = statcast(start_dt=start_date, end_dt=end_date)
    
    # Save to cache
    data.to_csv(CACHE_FILE, index=False)
    print(f"Data saved to {CACHE_FILE}")
    
    return data

def preprocess_data(data):
    """
    Preprocesses the data for the neural network.
    Creates 'is_hit' target and selects relevant features.
    """
    # Filter for relevant events that count as hits
    # events: single, double, triple, home_run are hits
    hit_events = ['single', 'double', 'triple', 'home_run']
    
    # Create target variable
    data['is_hit'] = data['events'].apply(lambda x: 1 if x in hit_events else 0)
    
    # Select features
    # We need features that represent the batter and pitcher context.
    # For a simple model, we might use:
    # - release_speed (pitch velocity)
    # - release_spin_rate
    # - p_throws (pitcher handedness)
    # - stand (batter handedness)
    # - batter (id) - maybe too sparse for a simple model without embeddings
    # - pitcher (id)
    
    # For this specific request "stats of a pitcher and hitting stats of a list of batters",
    # we ideally want aggregate stats entering the match.
    # However, constructing rolling averages is complex.
    # We will try to train on raw pitch characteristics + batter/pitcher IDs (or simple features).
    
    # Let's keep it simple: Predict hit probability based on:
    # Pitcher: release_speed, release_spin_rate, p_throws
    # Batter: stand, avg, slg, iso
    
    # Calculate batter stats
    batter_stats = calculate_batter_stats(data)
    batter_stats.to_csv('batter_stats.csv')
    
    # Merge batter stats
    data = data.merge(batter_stats, on='batter', how='left')
    
    features = ['release_speed', 'release_spin_rate', 'p_throws', 'stand', 'avg', 'slg', 'iso', 'is_hit']
    df = data[features].dropna()
    
    # Encode categorical
    df['p_throws'] = df['p_throws'].map({'R': 0, 'L': 1})
    df['stand'] = df['stand'].map({'R': 0, 'L': 1})
    
    # Save processed data for training
    df.to_csv('processed_data.csv', index=False)
    return df

def get_player_id(name):
    try:
        # Handle names with multiple spaces (e.g. "Hyun Jin Ryu")
        parts = name.strip().split(' ')
        if len(parts) < 2:
            return None
        
        first = parts[0]
        last = ' '.join(parts[1:]) # Combine the rest as last name
        
        # Special case for "DJ LeMahieu" where DJ is first
        # Special case for "Hyun Jin Ryu" where Ryu is last
        
        data = playerid_lookup(last, first)
        if not data.empty:
            return data.iloc[0]['key_mlbam']
    except:
        return None
    return None

def get_pitcher_profile(name):
    # For simplicity, we will look up the player in our cached data first
    # If not found, we would ideally fetch their stats.
    # Here we will just return average stats from the cache if available, 
    # or defaults if not found (to avoid crashing).
    
    if not os.path.exists(CACHE_FILE):
        return None
        
    df = pd.read_csv(CACHE_FILE)
    
    # We need to map name to ID or just use the ID if we had it.
    # The cache has 'pitcher' (ID).
    pid = get_player_id(name)
    if pid:
        player_data = df[df['pitcher'] == pid]
        if not player_data.empty:
            return {
                'release_speed': player_data['release_speed'].mean(),
                'release_spin_rate': player_data['release_spin_rate'].mean(),
                'p_throws': 0 if player_data.iloc[0]['p_throws'] == 'R' else 1
            }
            
    # Fallback or fetch fresh (omitted for speed, returning average of all)
    print(f"Pitcher {name} not found in cache. Using league average.")
    return {
        'release_speed': df['release_speed'].mean(),
        'release_spin_rate': df['release_spin_rate'].mean(),
        'p_throws': 0 # Assume Righty
    }

def calculate_batter_stats(df):
    """
    Calculates AVG, SLG, ISO for each batter.
    """
    # Define events
    hit_events = ['single', 'double', 'triple', 'home_run']
    ab_events = hit_events + ['field_out', 'strikeout', 'force_out', 'grounded_into_double_play', 'fielders_choice']
    
    # Helper to calculate stats
    def get_stats(group):
        hits = group['events'].isin(hit_events).sum()
        ab = group['events'].isin(ab_events).sum()
        
        # Total Bases
        singles = (group['events'] == 'single').sum()
        doubles = (group['events'] == 'double').sum()
        triples = (group['events'] == 'triple').sum()
        hrs = (group['events'] == 'home_run').sum()
        tb = singles + 2*doubles + 3*triples + 4*hrs
        
        if ab == 0:
            return pd.Series({'avg': 0.0, 'slg': 0.0, 'iso': 0.0})
            
        avg = hits / ab
        slg = tb / ab
        iso = slg - avg
        
        return pd.Series({'avg': avg, 'slg': slg, 'iso': iso})

    stats = df.groupby('batter').apply(get_stats)
    return stats

def get_batter_profile(name):
    if not os.path.exists(CACHE_FILE) or not os.path.exists('batter_stats.csv'):
        return None
        
    df = pd.read_csv(CACHE_FILE)
    stats_df = pd.read_csv('batter_stats.csv')
    
    pid = get_player_id(name)
    
    if pid:
        # Get Handedness from raw data
        player_data = df[df['batter'] == pid]
        stand = 0
        if not player_data.empty:
            stand = 0 if player_data.iloc[0]['stand'] == 'R' else 1
            
        # Get Stats
        player_stats = stats_df[stats_df['batter'] == pid]
        if not player_stats.empty:
            return {
                'stand': stand,
                'avg': player_stats.iloc[0]['avg'],
                'slg': player_stats.iloc[0]['slg'],
                'iso': player_stats.iloc[0]['iso']
            }
            
    # Fallback
    print(f"Batter {name} not found in cache. Assuming League Average.")
    return {'stand': 0, 'avg': 0.240, 'slg': 0.400, 'iso': 0.160}


if __name__ == "__main__":
    df = fetch_data()
    print(f"Fetched {len(df)} rows.")
    processed = preprocess_data(df)
    print(f"Processed {len(processed)} rows.")
    print(processed.head())
