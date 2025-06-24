import datetime
import difflib
import json as json_module
import os
from pathlib import Path

import google.generativeai as genai
import gspread
import requests
import spotipy
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from google.oauth2.service_account import Credentials
from spotipy.oauth2 import SpotifyOAuth

from flask_session import Session

# --- APP SETUP ---
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# --- ENVIRONMENT and CONFIG ---
script_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=script_dir / ".env")

# CORS configuration
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS")
origins = []
if CORS_ALLOWED_ORIGINS:
    origins = [origin.strip() for origin in CORS_ALLOWED_ORIGINS.split(',')]
CORS(app, origins=origins)

LLM_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"  # TLD cache folder

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")
GOOGLE_SERVICE_ACCOUNT_JSON_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH")


# --- ERROR LOGGING ---
def log_and_show_error(message):
    """Logs an error message to the console and displays it in the Streamlit app."""
    print(f"Error: {message}")


# --- AUTHENTICATION ---
# Google Generative AI Configuration
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    llm_model = genai.GenerativeModel(LLM_MODEL_NAME)
except Exception as e:
    print(f"Google AI SDK configuration failed: {e}")

# Google Sheets Authentication
try:
    service_account_path = script_dir / GOOGLE_SERVICE_ACCOUNT_JSON_PATH
    creds = Credentials.from_service_account_file(str(service_account_path),
                                                  scopes=["https://www.googleapis.com/auth/spreadsheets",
                                                          "https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds)
    spreadsheet = gc.open(GOOGLE_SHEET_NAME)
    worksheet = spreadsheet.sheet1
except Exception as e:
    print(f"Google Sheets authentication failed: {e}")


# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login():
    sp_oauth = create_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    return jsonify({'auth_url': auth_url})


@app.route("/spotify_auth")
def callback():
    sp_oauth = create_spotify_oauth()
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session["token_info"] = token_info
    return "<script>window.close();</script>"


@app.route("/process_poster", methods=["POST"])
@limiter.limit("10 per minute")
def process_poster():
    if 'poster' not in request.files:
        return jsonify({"error": "No poster file found"}), 400

    file = request.files['poster']
    filename = file.filename
    image_bytes = file.read()

    festival_info = get_festival_info_from_poster(image_bytes, filename)
    if not festival_info:
        return jsonify({"error": "Could not extract festival info"}), 500

    return jsonify(festival_info)


@app.route("/get_artist_data", methods=["POST"])
@limiter.limit("30 per minute")
def get_artist_data():
    data = request.get_json()
    artists = data.get("artists")

    if not artists:
        return jsonify({"error": "Missing required data"}), 400

    token_info = get_token()
    if not token_info:
        return jsonify({"error": "User not authenticated"}), 401

    sp = spotipy.Spotify(auth=token_info['access_token'])

    artist_data_list = []
    for artist_name in artists:
        artist_data = get_artist_top_tracks(sp, artist_name)
        if artist_data:
            artist_data_list.append(artist_data)

    return jsonify(artist_data_list)


@app.route("/create_playlist", methods=["POST"])
@limiter.limit("15 per minute")
def create_playlist():
    data = request.get_json()
    festival_name = data.get("festival_name")
    festival_geo = data.get("festival_geo")
    festival_year = data.get("festival_year")
    track_ids = data.get("track_ids")
    user_ip = data.get("user_ip")

    if not all([festival_name, festival_geo, track_ids, user_ip]):
        return jsonify({"error": "Missing required data"}), 400

    token_info = get_token()
    if not token_info:
        return jsonify({"error": "User not authenticated"}), 401

    sp = spotipy.Spotify(auth=token_info['access_token'])
    user_id = sp.current_user()["id"]

    playlist_name = f"{festival_name} - {festival_geo}"
    if festival_year:
        playlist_name = f"{festival_name} {festival_year} - {festival_geo}"
    playlist_url = create_spotify_playlist(sp, user_id, playlist_name, track_ids)

    if playlist_url:
        user_geo = get_user_geo(user_ip)
        save_playlist_to_sheet(user_id, festival_name, festival_geo, festival_year, playlist_url, user_ip, user_geo)
        return jsonify({"playlist_url": playlist_url})
    else:
        return jsonify({"error": "Could not create playlist"}), 500


# --- HELPER FUNCTIONS ---
def create_spotify_oauth():
    return SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="playlist-modify-public playlist-modify-private user-library-read"
    )


def get_token():
    token_info = session.get("token_info", None)
    if not token_info:
        return None
    now = int(datetime.datetime.now().timestamp())
    is_expired = token_info['expires_at'] - now < 60
    if is_expired:
        sp_oauth = create_spotify_oauth()
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info
    return token_info


def get_festival_info_from_poster(image_bytes, filename):
    """
    Sends the poster image to an LLM to extract artist names, using a cache.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # Ensure cache directory exists
    # Use a safe version of the filename for the cache, e.g. by removing special chars or hashing
    # For simplicity here, we'll use the filename directly but this could be improved.
    cache_file_name = "".join(c if c.isalnum() or c in ('.', '_') else '_' for c in filename) + ".json"
    cache_file_path = CACHE_DIR / cache_file_name

    # Check cache first
    if cache_file_path.exists():
        try:
            with open(cache_file_path, 'r') as f:
                cached_artists = json_module.load(f)
            print(f"LLM Result: Loaded from cache for '{filename}' from {cache_file_path}")
            return cached_artists
        except Exception as e:
            print(f"LLM Cache: Error reading cache file {cache_file_path}: {e}. Fetching from LLM.")

    if not image_bytes:
        return []

    # Determine MIME type based on file extension
    file_extension = filename.split('.')[-1].lower()  # Use filename
    if file_extension in ["jpg", "jpeg"]:
        mime_type = "image/jpeg"
    elif file_extension in ["png"]:
        mime_type = "image/png"
    elif file_extension in ["heic", "heif"]:
        mime_type = "image/heic"  # Or "image/heif" - check what the LLM expects
    else:
        log_and_show_error(f"Unsupported file type for LLM processing: {file_extension}")
        return []

    image_parts = [{"mime_type": mime_type, "data": image_bytes}]
    prompt = """
    Analyze the provided image, which is a music festival poster.
    Extract the following information:
    1. The name of the festival.
    2. The location of the festival (e.g., city, country).
    3. All artist and band names visible on the poster.
    4. Try to guess the year of the festival.

    Return the information as a JSON object with the following structure:
    {
      "festival_name": "Name of the Festival",
      "festival_location": "City, Country",
      "festival_year": 2024,
      "artists": ["Artist One", "Band Two", "DJ Three"]
    }

    If any information cannot be found, return null for that field.
    If no artists are found, return an empty list for the "artists" field.
    """
    try:
        print(f"LLM Request - Model: {LLM_MODEL_NAME}, Prompt: '{prompt[:100]}...' (plus image)")  # Log part of prompt
        response = llm_model.generate_content([prompt, image_parts[0]])  # Simplified for single image
        print(f"LLM Response Text: {response.text}")  # Log model response
        # Refined parsing based on typical Gemini JSON responses
        # Gemini might return the JSON string within a larger text block, potentially with markdown
        # A common way it returns JSON is in a block like ```json ... ```
        if "```json" in response.text:
            json_str = response.text.split("```json")[1].split("```")[0].strip()
        else:
            json_str = response.text.strip()  # Assume it's a direct JSON string if no markdown

        festival_info = json_module.loads(json_str)  # Use aliased import
        if isinstance(festival_info, dict) and "artists" in festival_info and isinstance(festival_info["artists"],
                                                                                         list):
            # Save to cache
            try:
                with open(cache_file_path, 'w') as f:
                    json_module.dump(festival_info, f)
                print(f"LLM Result: Saved to cache for '{filename}' at {cache_file_path}")
            except Exception as e:
                print(f"LLM Cache: Error writing cache file {cache_file_path}: {e}")
            return festival_info
        else:
            log_and_show_error(f"LLM returned unexpected format for artist names: {festival_info}")
            return None
    except json_module.JSONDecodeError as e:  # Use aliased import
        log_and_show_error(f"Error decoding JSON from LLM response: {e}. Response text: {response.text}")
        return None
    except Exception as e:
        log_and_show_error(f"Error extracting artists from poster via LLM: {e}")
        print(f"LLM Error: {e}")  # Log error to console
        return None


def get_artist_top_tracks(sp, artist_name):
    """
    Gets top 3 tracks, genres, and followers for an artist using enhanced search logic.
    Returns a dict: {
        'artist_id': spotify_artist_id,
        'artist_name': spotify_artist_name,
        'genres': list_of_genres,
        'followers': follower_count,
        'tracks': list_of_track_dicts [{'id': track_id, 'name': track_name}]
    } or None if no suitable artist found.
    """
    try:
        # Query for top 3 potential artist matches
        results = sp.search(q='artist:' + artist_name, type='artist', limit=3)
        print(f"Spotify raw search results for '{artist_name}': {results}")

        items = results['artists']['items']
        if not items:
            return None  # Return None if no items

        best_match_artist_info = None
        highest_similarity_ratio = 0.0
        query_lower = artist_name.lower()

        print(f"Comparing query '{query_lower}' with Spotify results:")
        for item in items:
            spotify_name_lower = item['name'].lower()
            similarity = difflib.SequenceMatcher(None, query_lower, spotify_name_lower).ratio()
            print(f"  - Spotify: '{item['name']}' (lower: '{spotify_name_lower}'), Similarity: {similarity:.4f}")

            if similarity > highest_similarity_ratio:
                highest_similarity_ratio = similarity
                best_match_artist_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'similarity': similarity,
                    # Store the full item to access genres/followers later if it's the best match
                    'full_item_details': item
                }

        SIMILARITY_THRESHOLD = 0.85

        if best_match_artist_info and best_match_artist_info['similarity'] >= SIMILARITY_THRESHOLD:
            selected_artist_id = best_match_artist_info['id']
            selected_artist_spotify_name = best_match_artist_info['name']
            actual_similarity = best_match_artist_info['similarity']

            # Get full artist details for genres and followers from the selected match
            # We make a specific call to sp.artist() to get the most detailed info
            artist_details_response = sp.artist(selected_artist_id)
            genres = artist_details_response.get('genres', [])
            followers = artist_details_response.get('followers', {}).get('total')  # Can be None

            print(
                f"Selected artist: '{selected_artist_spotify_name}' (Genres: {genres}, Followers: {followers}) for query '{artist_name}' with similarity ratio {actual_similarity:.4f}")

            tracks_response = sp.artist_top_tracks(selected_artist_id)
            print(
                f"Spotify top tracks for {selected_artist_spotify_name} (ID: {selected_artist_id}): {tracks_response}")

            found_tracks_info = []
            for track in tracks_response['tracks'][:3]:
                found_tracks_info.append({
                    'id': track['id'],
                    'name': track['name'],
                })

            return {
                'artist_id': selected_artist_id,
                'artist_name': selected_artist_spotify_name,
                'genres': genres,
                'followers': followers,
                'tracks': found_tracks_info
            }
        else:
            if best_match_artist_info:
                warning_msg = f"Query '{artist_name}': Best Spotify match '{best_match_artist_info['name']}' (Similarity: {best_match_artist_info['similarity']:.2f}) was below threshold {SIMILARITY_THRESHOLD}. Skipping."
                print(warning_msg)
            else:
                return None

    except Exception as e:
        log_and_show_error(f"Error fetching tracks for {artist_name} from Spotify: {e}")
        print(f"Spotify API error for {artist_name}: {e}")
        return None


def create_spotify_playlist(sp, user_id, playlist_name, track_ids):
    """Creates a Spotify playlist and returns the playlist URL."""
    if not track_ids:
        print("No tracks to add to the playlist.")
        return None
    try:
        playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True)
        sp.playlist_add_items(playlist_id=playlist['id'], items=track_ids)
        print(f"Playlist '{playlist_name}' created successfully!")
        return playlist['external_urls']['spotify']
    except Exception as e:
        log_and_show_error(f"Error creating Spotify playlist: {e}")
        return None


def save_playlist_to_sheet(user_id, festival_name, festival_geo, festival_year, playlist_url, user_ip, user_geo):
    """Saves playlist information to the Google Sheet."""
    try:
        row = [
            user_id,
            festival_name,
            festival_year,
            festival_geo,
            playlist_url,
            datetime.datetime.now().isoformat(),
            user_ip,
            user_geo,
            datetime.datetime.now().strftime("%Y-%m-%d")
        ]
        worksheet.append_row(row)
        print(f"Saved playlist to Google Sheet: {row}")
    except Exception as e:
        log_and_show_error(f"Error saving playlist to Google Sheet: {e}")


def get_user_geo(ip_address):
    if ip_address == '127.0.0.1':
        return "Localhost"
    try:
        response = requests.get(f'http://ip-api.com/json/{ip_address}')
        data = response.json()
        if data.get('status') == 'success':
            return f"{data.get('city', 'N/A')}, {data.get('regionName', 'N/A')}, {data.get('country', 'N/A')}"
        else:
            return "Unknown"
    except Exception as e:
        print(f"Could not get user geo for IP {ip_address}: {e}")
        return "Unknown"


if __name__ == "__main__":
    app.run(debug=True, port=5001)
