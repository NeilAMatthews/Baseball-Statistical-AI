import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from data_loader import get_pitcher_profile, get_batter_profile
import os

MODEL_PATH = 'baseball_model.keras'
SCALER_PATH = 'scaler.pkl'

def optimize_lineup(pitcher_name, batter_names):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return "Model or Scaler not found. Please train the model first."

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    pitcher_stats = get_pitcher_profile(pitcher_name)
    if pitcher_stats is None:
        return f"Pitcher {pitcher_name} not found (or no data available)."
        
    lineup_data = []
    valid_batters = []
    
    print(f"Optimizing lineup against {pitcher_name}...")
    
    for batter in batter_names:
        batter_stats = get_batter_profile(batter)
        if batter_stats:
            # Combine stats
            # Order must match training data: release_speed, release_spin_rate, p_throws, stand, avg, slg, iso
            row = [
                pitcher_stats['release_speed'],
                pitcher_stats['release_spin_rate'],
                pitcher_stats['p_throws'],
                batter_stats['stand'],
                batter_stats['avg'],
                batter_stats['slg'],
                batter_stats['iso']
            ]
            lineup_data.append(row)
            valid_batters.append(batter)
        else:
            print(f"Batter {batter} not found.")
            
    if not lineup_data:
        return "No valid batters found."
        
    X = pd.DataFrame(lineup_data, columns=['release_speed', 'release_spin_rate', 'p_throws', 'stand', 'avg', 'slg', 'iso'])
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    
    results = []
    for i, batter in enumerate(valid_batters):
        results.append({
            'batter': batter,
            'predicted_hit_prob': float(predictions[i][0])
        })
        
    # Sort by probability
    results.sort(key=lambda x: x['predicted_hit_prob'], reverse=True)
    
    return results

if __name__ == "__main__":
    # Example
    pitcher = "Gerrit Cole"
    batters = ["Aaron Judge", "Juan Soto", "Giancarlo Stanton", "Anthony Volpe", "Gleyber Torres", "Alex Verdugo", "DJ LeMahieu", "Austin Wells", "Oswaldo Cabrera"]
    
    print(f"Batters: {batters}")
    best_lineup = optimize_lineup(pitcher, batters)
    
    if isinstance(best_lineup, str):
        print(best_lineup)
    else:
        print("\nBest Lineup (Sorted by Hit Probability):")
        for i, player in enumerate(best_lineup):
            print(f"{i+1}. {player['batter']} - {player['predicted_hit_prob']:.3f}")
