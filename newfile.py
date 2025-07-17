import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import bcrypt
import mysql.connector
import os
from google.oauth2 import id_token
from google.auth.transport import requests

# Database initialization
conn = mysql.connector.connect(
    host="localhost",
    user="Jash",
    password="jash123",
    database="login_signup"
)
cursor = conn.cursor()

model  = load_model(r"C:\Users\jashp\Downloads\emotion-based-music-main\emotion-based-music-main\model.h5")
label = np.load(r"C:\Users\jashp\Downloads\emotion-based-music-main\emotion-based-music-main\labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Your Google OAuth client ID
CLIENT_ID = '825455712190-s99mps715fc1i98d1iudr0gc53j2egl6.apps.googleusercontent.com'

def verify_google_token(token):
    try:
        # Verify and decode the token
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)

        # Check if the token is issued for this app
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Invalid token issuer.')

        return idinfo
    except ValueError as e:
        st.error(f'Error verifying Google ID token: {e}')
        return None

# Function to handle Google login
def google_login():
    st.title('Google Login')
    st.write('Click the button below to log in with your Google account.')

    # Button to initiate Google login
    if st.button('Login with Google'):
        # Prompt user to select Google account
        token = st.text_input('Enter Google ID token:')
        if token:
            # Verify the Google ID token
            idinfo = verify_google_token(token)
            if idinfo:
                st.success(f'Logged in as: {idinfo["email"]}')
                return idinfo
            else:
                st.error('Failed to verify Google ID token.')

    return None


# Function to signup
def signup(username, password):
    # Hash the password before storing it in the database
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        cursor.execute('''INSERT INTO user (username, password_hash) VALUES (%s, %s)''', (username, password_hash))
        conn.commit()
        return True
    except mysql.connector.IntegrityError:
        return False  # Username already exists

# Function to login
def login(username, password):
    cursor.execute('''SELECT password_hash FROM user WHERE username = %s''', (username,))
    result = cursor.fetchone()
    if result:
        stored_password_hash = result[0]
        # Check if the provided password matches the stored hashed password
        if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            return True
    return False

# Function for signup page
def signup_page():
    st.title("Signup Page")

    # Signup form
    signup_username = st.text_input("Username (Signup)")
    signup_password = st.text_input("Password (Signup)", type="password")

    if st.button("Signup"):
        if signup_username and signup_password:
            if signup(signup_username, signup_password):
                st.success("Signup successful! Please login.")
                return "login"
            else:
                st.error("Username already exists.")
    return None

# Function for login page
def login_page():
    st.title("Login Page")

    # Login form
    login_username = st.text_input("Username (Login)")
    login_password = st.text_input("Password (Login)", type="password")

    if st.button("Login"):
        if login_username and login_password:
            if login(login_username, login_password):
                st.success("Login successful! Redirecting to main app...")
                return "main_app"
            else:
                st.error("Invalid username or password.")
    return None

# Function for main app (emotion recommender)
def main_app():
    st.header("Emotion Based Music Recommender")

    class EmotionProcessor:
        def recv(self, frame):
            frm = frame.to_ndarray(format="bgr24")

            ##############################
            frm = cv2.flip(frm, 1)

            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            lst = []

            if res.face_landmarks:
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

                if res.left_hand_landmarks:
                    for i in res.left_hand_landmarks.landmark:
                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                if res.right_hand_landmarks:
                    for i in res.right_hand_landmarks.landmark:
                        lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                lst = np.array(lst).reshape(1,-1)

                pred = label[np.argmax(model.predict(lst))]

                print(pred)
                cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

                np.save("emotion.npy", np.array([pred]))


            drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                                    connection_drawing_spec=drawing.DrawingSpec(thickness=1))
            drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
            drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)


            ##############################

            return av.VideoFrame.from_ndarray(frm, format="bgr24")

    lang = st.text_input("Language")
    singer = st.text_input("Singer")

    if lang and singer:
        webrtc_streamer(key="key", desired_playing_state=True,
                    video_processor_factory=EmotionProcessor)

    if st.button("Recommend me songs"):
        emotion_data = np.load("emotion.npy")
        if emotion_data is not None and len(emotion_data) > 0:
            emotion = emotion_data[0]
            if emotion:
                url = f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}"
                webbrowser.open(url)
                np.save("emotion.npy", np.array([""]))
            else:
                st.warning("Please let me capture your emotion first")
        else:
            st.warning("Please let me capture your emotion first")





def main():


    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main_app()
    else:
        # Show signup page first
        page = signup_page()

        if page == "login":
            st.experimental_rerun()

        # Show login page
        page = login_page()

        if page == "main_app":
            st.session_state.logged_in = True
            main_app()

if __name__ == "__main__":
    main()
