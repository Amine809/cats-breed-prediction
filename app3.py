import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("cat_breed_model.h5")

# Preprocess image function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize pixel values to be between 0 and 1

# Streamlit UI
st.set_page_config(
    page_title="application de prédiction de race de chat",
    page_icon="🐱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            color: #333;
            background-color: #f8f8f8;
        }
        .st-bw {
            max-width: 900px;
            margin: 0 auto;
        }
        .st-cy {
            color: #0078ff;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# UI components
st.title("application de prédiction de race de chat")
st.sidebar.title("Options")

# Upload a cat image
uploaded_file = st.sidebar.file_uploader("Choisir une image d'un chat...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image and make predictions
    img_array = preprocess_image(uploaded_file)
    prediction = model.predict(img_array)

    # Get the predicted breed
    predicted_breed_index = np.argmax(prediction)
    breed_names = ["Abyssinian", "American Bobtail", "American Curl", "American Shorthair", "Bengal", 
                   "Birman", "Bombay", "British Shorthair", "Egyptian Mau", "Exotic Shorthair", 
                   "Maine Coon", "Manx", "Norwegian Forest", "Persian", "Ragdoll", "Russian Blue", 
                   "Scottish Fold", "Siamois", "Sphynx", "Turkish Angora"]
    predicted_breed = breed_names[predicted_breed_index]

    # Display the results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.success(f"Race prédite: {predicted_breed}")
else:
    st.image("images/manja-vitolic-gKXKBY-C-Dk-unsplash.jpg", use_column_width=True)
