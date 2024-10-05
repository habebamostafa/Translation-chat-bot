import streamlit as st
from googletrans import Translator
import random

# Initialize the translator
translator = Translator()

# Simulated chatbot responses for demonstration
chatbot_responses = {
    'greeting': ['Hello! How can I assist you today?', 'Hi there! What do you need help with?'],
    'help': ['I can help you with translations and answer your questions!', 'Feel free to ask me anything!'],
    'default': ['Sorry, I don’t understand that.', 'Can you rephrase your question?']
}

# Function to simulate a chatbot response
def get_chatbot_response(user_message):
    if "hello" in user_message.lower() or "hi" in user_message.lower():
        return random.choice(chatbot_responses['greeting'])
    elif "help" in user_message.lower():
        return random.choice(chatbot_responses['help'])
    else:
        return random.choice(chatbot_responses['default'])

# Function to translate text from English to Arabic
def translate_text_to_arabic(text):
    translated = translator.translate(text, src='en', dest='ar')
    return translated.text

# Streamlit app layout
st.title("Chatbot for Translation or dialoge")
st.write("You can enter your message below:")

user_input = st.text_input("Your message:")
if st.button("Send"):
    # Check if the input is a request for translation (e.g., starts with "translate")
    if user_input.strip().lower().startswith("translate "):
        phrase_to_translate = user_input[len("translate "):].strip()  # Remove the "translate " part
        try:
            # Translate user input to Arabic
            translated_input = translate_text_to_arabic(phrase_to_translate)
            # Display translation
            st.write("**Translation:**", translated_input)
        except Exception as e:
            st.write("An error occurred during translation:", str(e))
    else:
        # Get chatbot response
        chatbot_response = get_chatbot_response(user_input)

        # Display responses
        st.write("**You:**", user_input)
        st.write("**Chatbot:**", chatbot_response)
