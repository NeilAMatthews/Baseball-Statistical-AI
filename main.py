from optimizer import optimize_lineup
import sys

def main():
    print("Welcome to the Baseball Lineup Optimizer!")
    
    if len(sys.argv) > 2:
        # Command line mode
        pitcher = sys.argv[1]
        batters = [b.strip() for b in sys.argv[2].split(',')]
    else:
        # Interactive mode
        pitcher = input("Enter Pitcher Name (e.g. Gerrit Cole): ")
        batters_input = input("Enter Batter Names (comma separated, e.g. Aaron Judge, Juan Soto): ")
        batters = [b.strip() for b in batters_input.split(',')]
        
    print(f"\nOptimizing lineup against {pitcher}...")
    results = optimize_lineup(pitcher, batters)
    
    if isinstance(results, str):
        print(f"Error: {results}")
    else:
        print("\nOptimal Lineup:")
        for i, player in enumerate(results):
            print(f"{i+1}. {player['batter']} (Hit Prob: {player['predicted_hit_prob']:.3f})")

if __name__ == "__main__":
    main()
