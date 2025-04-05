 # Import necessary modules
import streamlit as st
import pyttsx3
import threading
import tweepy
from main import clean, clf, cv, extract_text



# Twitter API Bearer Token
bearer_token = 'AAAAAAAAAAAAAAAAAAAAABGuwwEAAAAAvAn3ByVMr4UIYH9PGBW0OFWFbQk%3DGtO5yRzIjZq8LfYDjggLGbfYCVZbejffleOfii8NDFcKqaWJZD'
client = tweepy.Client(bearer_token=bearer_token)

# Set Streamlit page config
st.set_page_config(page_title="Hate Speech Detection", page_icon="🤫", layout="wide")


# Title and description
st.write("### A tool to detect hate speech, offensive language, or neutral content.")

# Sidebar with input options
# input_choice = st.sidebar.radio("Select Input Type:", ["Text Input", "Twitter URL", "Upload Image"])
input_choice = st.sidebar.radio("Select Input Type:", ["Text Input", "Twitter URL", "Upload Image"], key="input_choice")
st.sidebar.write("Choose an input type, enter your text, or upload an image to analyze hate speech.")

# Function to speak text asynchronously
def speak_async(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to run speech asynchronously in a thread
def speak_async_thread(text):
    threading.Thread(target=speak_async, args=(text,)).start()

def find_verdict(user_input):
    cleaned_input = clean(user_input)
    input_data = cv.transform([cleaned_input]).toarray()
    verdict = clf.predict(input_data)[0]
    st.success(f"**Verdict**: {verdict}")
    speak_async_thread(f"The Verdict is {verdict}")

# For text input
if st.session_state.input_choice == "Text Input":
    user_input = st.text_area("Enter your text:", help="Type text you want to analyze.")
    if st.button("Detect Hate Speech"):
        if user_input:
            find_verdict(user_input)
        else:
            st.warning("Please enter text before clicking the button.")

# For Twitter URL
elif input_choice == 'Twitter URL':
    twitter_url = st.text_input("Enter Twitter URL:", help="Paste the URL of the tweet here.")
    if st.button("Fetch Tweet and Detect"):
        if twitter_url:
            try:
                tweet_id = twitter_url.split("/")[-1]
                tweet = client.get_tweet(tweet_id, tweet_fields=["text"])
                tweet_content = tweet.data["text"]
                find_verdict(tweet_content)
            except tweepy.TweepyException as e:
                st.error(f"Error fetching tweet: {str(e)}")
        else:
            st.warning("Please enter URL before clicking the button.")

# For image input
elif input_choice == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        extracted_text = extract_text(uploaded_file)
        st.write("**Extracted Text:**")
        st.write(extracted_text)
        find_verdict(extracted_text)
