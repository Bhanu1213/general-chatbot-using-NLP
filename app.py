import csv
import datetime
import json
import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load the intent file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Ensure the chat log file exists
def ensure_chat_log_exists():
    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

# Function to save chat history
def save_chat_history(user_input, response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([user_input, response, timestamp])

# Main function to handle the Streamlit app
def main():
    st.set_page_config(page_title="Chatbot Intents", page_icon="", layout="wide")
    st.title("Chatbot using NLP")
    st.image("chatbot.jpeg", width=700)

    # Sidebar menu options
    menu = ['Home', "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Initialize session state variables
    if 'counter' not in st.session_state:
        st.session_state.counter = 0

    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press enter to start.")
        ensure_chat_log_exists()

        # User input
        user_input = st.text_input("You:", key=f"user_input_{st.session_state.counter}")

        if user_input:
            st.session_state.counter += 1
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_{st.session_state.counter}")
            
            # Save the conversation
            save_chat_history(user_input, response)

            # Handle termination message
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me, have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        with open("chat_log.csv", 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to human language.")
        st.write("""
                 The project is divided into two parts:
                 1. NLP technologies and logistic regression algorithm are used to train the chatbot.
                 2. The chatbot interface is built using Streamlit to create a simple web interface.
                 """)
        st.subheader("Dataset")
        st.write("""
            The dataset used in this project is a collection of intents and entities.
            - **Intents**: The intents are predefined categories that help the chatbot understand user input.
            - **Entities**: The entities are specific data points that are extracted from the user input.
            - **Text**: The input entered by the user.
        """)
        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface is built using Streamlit, providing a text-based communication.")
        st.subheader("Conclusion:")
        st.write("In this project, a chatbot is created to understand and respond to user inputs effectively using NLP and logistic regression.")

# Run the app
if __name__ == "__main__":
    main()
