import gradio as gr
import pandas as pd
import joblib
import os
import spotipy
import pylast
import discogs_client
from spotipy.oauth2 import SpotifyClientCredentials
from queue import PriorityQueue
from fuzzywuzzy import fuzz

final_model = joblib.load('final_model.pkl')
# Set up authentication with the Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ['SPOT_API'], client_secret=os.environ['SPOT_SECRET']))
network = pylast.LastFMNetwork(api_key=os.environ['LAST_API'], api_secret=os.environ['LAST_SECRET'])
d = discogs_client.Client('app/0.1', user_token=os.environ['DIS_TOKEN'])
genre_list = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient',
       'anime', 'black-metal', 'bluegrass', 'blues', 'brazil',
       'breakbeat', 'british', 'cantopop', 'chicago-house', 'children',
       'chill', 'classical', 'club', 'comedy', 'country', 'dance',
       'dancehall', 'death-metal', 'deep-house', 'detroit-techno',
       'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm',
       'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk',
       'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove',
       'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle',
       'heavy-metal', 'hip-hop', 'honky-tonk', 'house', 'idm', 'indian',
       'indie-pop', 'indie', 'industrial', 'iranian', 'j-dance', 'j-idol',
       'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino',
       'malay', 'mandopop', 'metal', 'metalcore', 'minimal-techno', 'mpb',
       'new-age', 'opera', 'pagode', 'party', 'piano', 'pop-film', 'pop',
       'power-pop', 'progressive-house', 'psych-rock', 'punk-rock',
       'punk', 'r-n-b', 'reggae', 'reggaeton', 'rock-n-roll', 'rock',
       'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo',
       'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'soul',
       'spanish', 'study', 'swedish', 'synth-pop', 'tango', 'techno',
       'trance', 'trip-hop', 'turkish', 'world-music']





def get_track_genre(track_id,artist_name,track_name):
    genres = {}
    track_spot = sp.track(track_id)
    artist = sp.artist(track_spot['artists'][0]['external_urls']['spotify'])
    album_id = track_spot['album']['id']
    album = sp.album(album_id)
    genres.update({genre: 100 for genre in album['genres']})
    genres.update({genre: 100 for genre in artist['genres']})

    try:
        if network.get_track(artist_name, track_name):
            track_last = network.get_track(artist_name, track_name)
            top_tags = track_last.get_top_tags(limit=5)
            tags_list = {tag.item.get_name(): int(tag.weight) for tag in top_tags}
            genres.update(tags_list)
    except pylast.WSError as e:
        if str(e) == "Track not found":
            # Handle the error here
            pass

    results = d.search(track_name, artist=artist_name, type='release')
    if results:
        release = results[0]
        if release.genres:
            genres.update({genre: 50 for genre in release.genres})
        if release.styles:
            genres.update({genre: 50 for genre in release.styles})


    print(genres)
    return genres


def similar(genre1, genre2):
    score = fuzz.token_set_ratio(genre1, genre2)
    return genre1 if score >85 else None

def find_genre(genres, scraped_genres):
    pq = PriorityQueue()
    for genre, weight in scraped_genres.items():
        pq.put((-weight, genre))
    while not pq.empty():
        weight, genre = pq.get()
        if genre in genres:
            return genre
        else:
            for g in genres:
                if similar(g, genre):
                    return g
    return None


def match_genres_to_list(track_id,artist_name,track_name):
    track_genres=get_track_genre(track_id,artist_name,track_name)
    return find_genre(genre_list,track_genres)

def search_songs(query):
    results = sp.search(q=query, type="track")
    songs = [f"{index}. {item['name']} by {item['artists'][0]['name']}" for index, item in enumerate(results["tracks"]["items"])]

    track_ids = [item["id"] for item in results["tracks"]["items"]]
    return songs, track_ids


def get_song_features(song, track_ids):
    index = int(song.split(".")[0])
    track_id = track_ids[index]
    track_info = sp.track(track_id)
    artist_name = track_info['artists'][0]['name']
    track_name = track_info['name']
    features = sp.audio_features([track_id])[0]
    genre = match_genres_to_list(track_id,artist_name,track_name)
    key_map = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
    key = str(key_map[features['key']])
    mode_map = { 1: "Major", 0: "Minor"}
    mode = mode_map[features['mode']]
    
    explicit_real = track_info['explicit']
    features_list = [
        features['duration_ms'],
        explicit_real,
        features['danceability'],
        features['energy'],
        key,
        features['loudness'],
        mode,
        features['speechiness'],
        features['acousticness'],
        features['instrumentalness'],
        features['liveness'],
        features['valence'],
        features['tempo'],
        str(features['time_signature']),
        genre
    ]
    
    return features_list

theme = gr.themes.Monochrome(
    # text_size="text_lg",
    font=[gr.themes.GoogleFont('Neucha'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
)
with gr.Blocks(theme=theme,css = "@media (max-width: 600px) {" +
        ".gradio-container { flex-direction: column;}" +
        ".gradio-container h1 {font-size: 30px !important ;margin-left: 20px !important; line-height: 30px !important}" +
        ".gradio-container h2 {font-size: 15px !important;margin-left: 20px !important;margin-top: 20px !important;}"+
        ".gradio-container img{width : 100px; height : 100px}}") as demo:
    with gr.Row():
        image = gr.HTML("<div style='display: flex; align-items: center;'><img src='file=images/cat-jam.gif' alt='My gif' width='200' height='200'>" +
                        "<div><h1 style='font-size: 60px; line-height: 24px; margin-left: 50px;'>Music Popularity Prediction</h1>" +
                        "<h2 style='font-size: 24px; line-height: 18px; margin-left: 50px; margin-top: 50px'>by Keh Zheng Xian</h2></div></div>")

    with gr.Row():
        with gr.Column():
            search_box = gr.Textbox(label="Search for songs")
            song_dropdown = gr.Dropdown(label="Select a song", choices=[])
            # features_box = gr.Textbox(label="Song features", interactive=False)
            inputs = [
                gr.Number(label="duration_ms",interactive=True),
                gr.Checkbox(label="explicit",interactive=True),
                gr.Slider(0.0, 1.0, label="danceability",interactive=True),
                gr.Slider(0.0, 1.0, label="energy",interactive=True),
                gr.Dropdown(label="key", choices=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],interactive=True),
                gr.Number(label="loudness",interactive=True),
                gr.Radio(label="mode", choices=["Major", "Minor"],interactive=True),
                gr.Slider(0.0, 1.0, label="speechiness",interactive=True),
                gr.Slider(0.0, 1.0, label="acousticness",interactive=True),
                gr.Slider(0.0, 1.0, label="instrumentalness",interactive=True),
                gr.Slider(0.0, 1.0, label="liveness",interactive=True),
                gr.Slider(0.0, 1.0, label="valence",interactive=True),
                gr.Number(label="tempo",interactive=True),
                gr.Dropdown(label="time_signature", choices=[3, 4, 5, 6, 7],interactive=True),
                gr.Dropdown(label="track_genre", choices=genre_list,interactive=True)
            ]
            predict_button = gr.Button(label="Predict popularity")

        with gr.Column():
            popularity_box = gr.HTML("<div style='display: flex; align-items: center;'><img src='file=images/pepe-waiting.gif' alt='My gif 2' width='200' height='200'>" +
                        "<div><h1 style='font-size: 30px; line-height: 24px; margin-left: 50px;'>Waiting for your song...</h1></div>",elem_id="output")
    track_ids_var = gr.State()
    def update_dropdown(query,track_ids):
        songs, track_ids = search_songs(query)
        return {song_dropdown: gr.update(choices=songs), track_ids_var: track_ids}

    search_box.change(fn=update_dropdown, inputs=[search_box,track_ids_var], outputs=[song_dropdown,track_ids_var])

    def update_features(song,track_ids):
        features = get_song_features(song, track_ids)
        return features

    def predict_popularity(duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature,track_genre):
        # Convert the key input from a string to an integer value
        key_map = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
        key_real = str(key_map[key])
        
        explicit_real = int(explicit)
        # Convert the mode input from a string to an integer value
        mode_map = {"Major": 1, "Minor": 0}
        mode_real = mode_map[mode]

        data = {
            "duration_ms": [duration_ms],
            "explicit": [explicit_real],
            "danceability": [danceability],
            "energy": [energy],
            "key": [key_real],
            "loudness": [loudness],
            "mode": [mode_real],
            "speechiness": [speechiness],
            "acousticness": [acousticness],
            "instrumentalness": [instrumentalness],
            "liveness": [liveness],
            "valence": [valence],
            "tempo": [tempo],
            "time_signature": [str(time_signature)],
            "track_genre": [track_genre]
        }

        df = pd.DataFrame(data)
        print(df)
        print(final_model.predict(df))
        # Use your trained model to predict popularity based on the input features
        if(final_model.predict(df)[0] == 1):
            return ("<div style='display: flex; align-items: center;'><img src='file=images/pepe-jam.gif' alt='My gif 3' width='200' height='200'>" +
                        "<div><h1 style='font-size: 30px; line-height: 24px; margin-left: 50px;'>Your song issa boppp</h1></div>")
        else:
            return ("<div style='display: flex; align-items: center;'><img src='file=images/pepo-sad-pepe.gif' alt='My gif 4' width='200' height='200'>" +
                        "<div><h1 style='font-size: 30px; line-height: 24px; margin-left: 50px;'>Not a bop....</h1></div>")

    song_dropdown.change(fn=update_features, inputs=[song_dropdown,track_ids_var], outputs=inputs)
    predict_button.click(fn=predict_popularity, inputs=inputs, outputs=popularity_box, scroll_to_output=True,
                         _js="const element = document.querySelector('output');"+
                             "const rect = element.getBoundingClientRect();"+
                             "const options = {left: rect.left, top: rect.top, behavior: 'smooth'}"+
                             "parentIFrame' in window ?"
                             "window.parentIFrame.scrollTo(options):"+
                             "window.scrollTo(options)")

    demo.launch()
