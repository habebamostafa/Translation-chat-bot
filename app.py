import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import PyPDF2
import fitz
from tensorflow.keras import layers, models

# Create a dummy chatbot model
def create_dummy_chatbot_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(100,)))  # Example input shape
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # Example output layer
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Save the model as a .h5 file
dummy_chatbot_model = create_dummy_chatbot_model()
dummy_chatbot_model.save('chatbot_model.h5')

# Create a dummy translation model
def create_dummy_translation_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(100,)))  # Example input shape
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # Example output layer
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Save the model as a .h5 file
dummy_translation_model = create_dummy_translation_model()
dummy_translation_model.save('translation_model.h5')

def read_pdf(file):
    text = ""
    # Use BytesIO to read the uploaded file
    with fitz.open(stream=file.read(), filetype='pdf') as pdf_document:
        for page in pdf_document:
            text += page.get_text()
    return text

# Load your models
def load_translation_model():
    # Load your translation model here
    return tf.keras.models.load_model('/workspaces/Translation-chat-bot/translation_model.h5')

def load_chat_model():
    # Load your chatbot model here
    return tf.keras.models.load_model('/workspaces/Translation-chat-bot/chatbot_model.h5')

def load_image_to_text_model():
    return tf.keras.models.load_model('cnn_model.h5')



def image_to_text(image, model):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(gray_image, (32, 32)) / 255.0
    # Add a channel dimension to the input image
    input_image = np.expand_dims(resized_image, axis=-1)  # Now shape is (32, 32, 1)
    input_image = np.expand_dims(input_image, axis=0)  # Now shape is (1, 32, 32, 1)

    # If your model expects 3 channels, you can repeat the gray image across the 3rd dimension
    input_image_rgb = np.repeat(input_image, 3, axis=-1)  # Shape becomes (1, 32, 32, 3)

    prediction = model.predict(input_image_rgb)
    return "Extracted Text from Image"

# Predefined image paths (local files or URLs)
predefined_images = {
    "Sample Image 1": "/workspaces/Translation-chat-bot/Screenshot 2024-10-06 161450.png",
    "Sample Image 2": "/workspaces/Translation-chat-bot/Screenshot 2024-10-06 161505.png",
}

# Image-to-text related functions (from your code)
mapping_inverse={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'}
def convert_2_gray(image):
    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Check if the image has at least 2 dimensions
    if len(image_np.shape) < 2:
        raise ValueError("Input image is not valid. It should have at least 2 dimensions.")

    # Handle cases for images with different channel counts
    if len(image_np.shape) == 3:
        # Check if the image has an alpha channel (4 channels)
        if image_np.shape[2] == 4:
            # Convert RGBA to RGB by removing the alpha channel
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        # Convert RGB to grayscale
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    elif len(image_np.shape) == 2:
        # The image is already grayscale
        gray_image = image_np
    else:
        raise ValueError("Unexpected image format.")

    return gray_image

def binarization(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return thresh

def dilate(image, words=False):
    m, n = 3, 1
    if words:
        m = n = 6
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, m))
    return cv2.dilate(image, rect_kernel, iterations=3)

def find_rect(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    return sorted(rects, key=lambda x: x[0])

def extract(image):
    model = load_image_to_text_model()  # Load the image-to-text model
    chars = []
    image_cpy = convert_2_gray(image)
    bin_img = binarization(image_cpy)
    full_dil_img = dilate(bin_img, words=True)
    words = find_rect(full_dil_img)

    prev_x_end = 0
    for word in words:
        x, y, w, h = word
        img = image_cpy[y:y + h, x:x + w]
        if x - prev_x_end > 20:
            chars.append(' ')
        prev_x_end = x + w

        bin_img = binarization(convert_2_gray(img))
        dil_img = dilate(bin_img)
        char_parts = find_rect(dil_img)

        for char in char_parts:
            cx, cy, cw, ch = char
            ch_img = img[cy:cy + ch, cx:cx + cw]
            empty_img = np.full((32, 32, 1), 255, dtype=np.uint8)
            resized = cv2.resize(ch_img, (16, 22), interpolation=cv2.INTER_CUBIC)
            gray = convert_2_gray(resized)
            empty_img[3:3 + 22, 3:3 + 16, 0] = gray

            gray_rgb = cv2.cvtColor(empty_img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            prediction = model.predict(np.array([gray_rgb]), verbose=0)
            predicted_char = mapping_inverse[np.argmax(prediction)]
            chars.append(predicted_char)

    return ''.join(chars)


# Load models
translation_model = load_translation_model()
chat_model = load_chat_model()
image_to_text_model = load_image_to_text_model()

# Streamlit interface
st.title("Multimodal Translation and Chatbot App")

option = st.selectbox("Choose Input Type", ["Text Translation", "Chat", "Image to Text", "PDF Translation"])

if option == "Text Translation":
    input_text = st.text_input("Enter text for translation")
    if st.button("Translate"):
        # translated = translate_text(input_text, translation_model)
        st.write('translated')

if option == "Chat":
    # Initialize chat history if not already done
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    input_text = st.text_input("Talk to the chatbot")

    if st.button("Chat"):
        # Simulate a chat response (replace this with your actual chat model prediction)
        response = "This is a simulated response."  # Placeholder for chat model response

        # Store user input and response in chat history
        st.session_state.chat_history.append(("User", input_text))
        st.session_state.chat_history.append(("Chatbot", response))

    # Display chat history
    st.write("### Chat History")
    chat_box = st.empty()
    for speaker, message in st.session_state.chat_history:
        chat_box.write(f"**{speaker}:** {message}")
  
elif option == "Image to Text":
    st.write("Select an option to extract text from an image:")

    upload_option = st.radio("Choose Input Method", ("Upload Image", "Try Sample Image"))

    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Extract Text from Uploaded Image"):
                extracted_text = extract(image)
                st.write(extracted_text)

                # Optionally, allow translation of the extracted text
                if st.button("Translate Extracted Text"):
                    # translated = translate_text(extracted_text, translation_model)
                    st.write('translated')

    elif upload_option == "Try Sample Image":
        st.write("Select a predefined image to extract text from:")

        # Let the user choose from predefined images
        selected_image = st.selectbox("Choose an image", list(predefined_images.keys()))
        image_path = predefined_images[selected_image]

        # Load and display the selected image
        image = Image.open(image_path)
        st.image(image, caption="Selected Image", use_column_width=True)

        if st.button("Extract Text from Sample Image"):
            extracted_text = extract(image)
            st.write(extracted_text)

            # Optionally, allow translation of the extracted text
            if st.button("Translate Extracted Text"):
                # translated = translate_text(extracted_text, translation_model)
                st.write('translated')

elif option == "PDF Translation":
    uploaded_pdf = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_pdf is not None:
        # Read PDF content
        pdf_text = read_pdf(uploaded_pdf)
        st.write("Original Text:")
        st.write(pdf_text)

        if st.button("Translate PDF Text"):
            # Placeholder for translation functionality
            st.write('Translated text will appear here.')
