# This code is a modified version of a script found at https://github.com/jackwu502/NSVA/blob/main/tools/download_video_by_gameid_eventid_date.py

import os
import json
import requests
import pandas as pd
from random import randint
from zipfile import ZipFile
import time
from datetime import datetime

# List of user agents for headers
USER_AGENTS = [
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
]

def format_date(game_date):
    """
    Converts game_date to 'YYYY/MM/DD' format.
    """
    for fmt in ('%m/%d/%Y', '%Y/%m/%d', '%Y-%m-%d', '%m-%d-%Y'):
        try:
            return datetime.strptime(game_date, fmt).strftime('%Y/%m/%d')
        except ValueError:
            continue
    raise ValueError(f"Date format for '{game_date}' is not recognized.")

def sanitize_filename(s):
    """
    Sanitizes a string to be safe for filenames.
    Removes or replaces characters that are invalid in filenames.
    """
    return "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in s).strip()

def download_video(game_id, event_id, date, result, output_dir):
    metadata_url = f'https://stats.nba.com/stats/videoeventsasset?GameEventID={event_id}&GameID={game_id}'
    random_agent = USER_AGENTS[randint(0, len(USER_AGENTS) - 1)]

    # Fetch UUID metadata
    try:
        r = requests.get(
            metadata_url,
            headers={
                'Accept': 'application/json, text/plain, */*',
                'Origin': 'https://www.nba.com',
                'Host': 'stats.nba.com',
                'User-Agent': random_agent,
                'Referer': 'https://www.nba.com/',
            },
            timeout=10  # Increased timeout for reliability
        )
        r.raise_for_status()
        json_data = r.json()
        uuid = json_data['resultSets']['Meta']['videoUrls'][0]['uuid']
    except (requests.exceptions.RequestException, IndexError, json.JSONDecodeError) as e:
        print(f"Failed to fetch UUID for GameID {game_id}, EventID {event_id}: {e}")
        return None

    # Construct video URL and download
    video_url = f'https://videos.nba.com/nba/pbp/media/{date}/{game_id}/{event_id}/{uuid}_1280x720.mp4'
    print(f"Constructed video URL: {video_url}")
    try:
        video_response = requests.get(
            video_url,
            headers={
                'User-Agent': random_agent,
                'Referer': 'https://www.nba.com/',
                'Origin': 'https://www.nba.com',
            },
            allow_redirects=True,
            timeout=10  # Increased timeout for reliability
        )
        if video_response.status_code == 200 and len(video_response.content) > 1000:
            # Sanitize the RESULT string for filename
            sanitized_result = sanitize_filename(result)
            filename = f"{game_id}-{event_id}-{sanitized_result}.mp4"
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'wb') as video_file:
                video_file.write(video_response.content)
            print(f"Downloaded: {filename}")
            return file_path
        else:
            print(f"Failed to download or invalid content for {game_id}-{event_id}. Content size: {len(video_response.content)} bytes. Status Code: {video_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download video for {game_id}-{event_id}: {e}")
    return None

# Main function to loop over events and create a zip of videos
def download_videos_to_zip(csv_file='nba_shots_data.csv', zip_filename='nba_shots_videos.zip'):
    # Determine the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define full paths for CSV and zip file
    csv_path = os.path.join(script_dir, csv_file)
    zip_path = os.path.join(script_dir, zip_filename)
    output_dir = os.path.join(script_dir, "temp_videos")
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV and specify GAME_ID as a string
    try:
        df = pd.read_csv(csv_path, dtype={'GAME_ID': str})
    except FileNotFoundError:
        print(f"ERROR: The file {csv_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"ERROR: The file {csv_path} is empty.")
        return
    except pd.errors.ParserError as e:
        print(f"ERROR: Failed to parse CSV file: {e}")
        return

    # Create zip file in the script directory
    with ZipFile(zip_path, 'w') as zipf:
        for idx, row in df.iterrows():
            game_id = row.get('GAME_ID')
            event_id = row.get('EVENTNUM')
            game_date_raw = row.get('GAME_DATE')
            result = row.get('RESULT')  # Get the 'RESULT' field

            if not game_id or not event_id or not game_date_raw:
                print(f"Skipping row {idx}: Missing GAME_ID, EVENTNUM, or GAME_DATE.")
                continue

            if pd.isna(result):
                print(f"Skipping row {idx}: Missing RESULT.")
                continue

            try:
                date = format_date(game_date_raw)
            except ValueError as ve:
                print(f"Skipping row {idx}: {ve}")
                continue

            print(f"Processing GameID: {game_id}, EventID: {event_id}, Date: {date}, Result: {result}")
            video_path = download_video(game_id, event_id, date, result, output_dir)

            if video_path:
                try:
                    zipf.write(video_path, os.path.basename(video_path))
                    print(f"Added to zip: {os.path.basename(video_path)}")
                    os.remove(video_path)
                except Exception as e:
                    print(f"Failed to add {video_path} to zip: {e}")
            else:
                print(f"Video not downloaded for {game_id}-{event_id}.")

            # Wait to avoid rate limits
            print("Waiting 1 second to avoid rate limits...")
            time.sleep(1)

    # Remove temporary folder
    try:
        os.rmdir(output_dir)
        print("Removed temporary video directory.")
    except OSError as e:
        print(f"Could not remove temporary directory: {e}")

    print(f"All videos processed and zipped successfully. Zip file saved at: {zip_path}")

# Run main function
if __name__ == "__main__":
    download_videos_to_zip()

