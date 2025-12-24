import pandas as pd
from pybaseball import playerid_lookup
import os

CACHE_FILE = 'statcast_2024_sample.csv'

def debug_lookup(last, first):
    print(f"Looking up {first} {last}...")
    try:
        data = playerid_lookup(last, first)
        if not data.empty:
            pid = data.iloc[0]['key_mlbam']
            print(f"Found ID: {pid}")
            return pid
        else:
            print("Player not found in lookup.")
            return None
    except Exception as e:
        print(f"Lookup error: {e}")
        return None

def check_cache(pid, role='batter'):
    if not os.path.exists(CACHE_FILE):
        print("Cache file not found.")
        return

    df = pd.read_csv(CACHE_FILE)
    print(f"Cache size: {len(df)} rows")
    
    if pid:
        count = df[df[role] == pid].shape[0]
        print(f"ID {pid} found {count} times in {role} column.")
    else:
        print("No ID to check.")

if __name__ == "__main__":
    # Check Aaron Judge
    judge_id = debug_lookup("Judge", "Aaron")
    check_cache(judge_id, 'batter')
    
    # Check Gerrit Cole
    cole_id = debug_lookup("Cole", "Gerrit")
    check_cache(cole_id, 'pitcher')
    
    # Check Juan Soto
    soto_id = debug_lookup("Soto", "Juan")
    check_cache(soto_id, 'batter')
