import pandas as pd
import os
from nba_api.stats.endpoints import playbyplay, leaguegamefinder
from enum import Enum
import re

class EventMsgType(Enum):
    FIELD_GOAL_MADE = 1
    FIELD_GOAL_MISSED = 2
    FREE_THROW_ATTEMPT = 3

def get_all_games(season='2023-24', season_type='Regular Season'):
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type
    )
    games = gamefinder.get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE']).dt.strftime('%Y/%m/%d')  # Format date
    return games[['GAME_ID', 'GAME_DATE']]  # Get game IDs and formatted dates

def categorize_result(home_desc, visitor_desc):
    combined_desc = f"{home_desc or ''} {visitor_desc or ''}".strip()
    if re.search(r'MISS', combined_desc, re.IGNORECASE):
        return 'MISS'
    elif re.search(r'3PT', combined_desc, re.IGNORECASE):
        return 'THREE POINTER'
    elif re.search(r'Free Throw', combined_desc, re.IGNORECASE):
        return 'FT MADE'
    else:
        return 'TWO POINTER'

def fetch_shot_events(game_id, game_date):
    play_by_play_df = playbyplay.PlayByPlay(game_id).get_data_frames()[0]
    # Filter for specific event types: field goal made, field goal missed, or free throw attempt
    shots_df = play_by_play_df[play_by_play_df['EVENTMSGTYPE'].isin([
        EventMsgType.FIELD_GOAL_MADE.value,
        EventMsgType.FIELD_GOAL_MISSED.value,
        EventMsgType.FREE_THROW_ATTEMPT.value
    ])].copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    # Apply categorize_result function and add GAME_DATE
    shots_df['RESULT'] = shots_df.apply(lambda row: categorize_result(row['HOMEDESCRIPTION'], row['VISITORDESCRIPTION']), axis=1)
    shots_df['GAME_DATE'] = game_date  # Add game date as a new column
    print("DEBUG: fetch_shot_events DataFrame:\n", shots_df.head())  # Debugging line to confirm 'GAME_DATE' and 'RESULT' are present
    
    return shots_df[['GAME_ID', 'EVENTNUM', 'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE', 
                     'HOMEDESCRIPTION', 'VISITORDESCRIPTION', 'SCORE', 'GAME_DATE', 'RESULT']]

def collect_shots(season='2023-24', target_shot_count=500):
    all_games = get_all_games(season)
    all_games = all_games.sample(frac=1).reset_index(drop=True)  # Shuffle games for random selection
    all_shots = pd.DataFrame()
    seen_game_ids = set()
    
    for _, game in all_games.iterrows():
        game_id = game['GAME_ID']
        game_date = game['GAME_DATE']
        
        if game_id in seen_game_ids:
            continue
        seen_game_ids.add(game_id)
        
        shots_df = fetch_shot_events(game_id, game_date)
        all_shots = pd.concat([all_shots, shots_df], ignore_index=True)
        
        print(f"Collected {len(shots_df)} shots from game {game_id} on {game_date}. Total shots collected: {len(all_shots)}")
        
        if len(all_shots) >= target_shot_count:
            break
    
    all_shots = all_shots.head(target_shot_count)
    all_shots['ID'] = range(1, len(all_shots) + 1)  # Assign ascending ID starting from 1
    
    # Debugging line to check final columns before saving
    print("DEBUG: Final all_shots DataFrame before saving:\n", all_shots.head())
    print("Columns in all_shots before saving to CSV:", all_shots.columns.tolist())
    
    return all_shots

def save_to_csv(shots_df, filename='nba_shots_data.csv'):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    shots_df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

# Main script
if __name__ == "__main__":
    target_shot_count = 50  # Adjust based on your data needs

    shots_df = collect_shots(target_shot_count=target_shot_count)
    if shots_df is not None:
        save_to_csv(shots_df)
    else:
        print("No shot data found.")
