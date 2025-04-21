# app.py
import streamlit as st
import random

# Random words to build dummy strings from
word_pool = [
    "apple", "banana", "orange", "table", "chair", "window",
    "book", "river", "cloud", "mountain", "sun", "moon",
    "light", "shadow", "ocean", "forest", "sky", "tree",
    "stone", "fire", "flower", "dream", "mirror", "path"
]

# Dummy model placeholder
class DummyModel:
    def eval(self):
        pass

    def __call__(self, text_input):
        # Generate 10 random sentences based on the input
        return [" ".join(random.choices(word_pool, k=random.randint(5, 10))) for _ in range(10)]

# Load model
model = DummyModel()
model.eval()

# Streamlit interface
st.title("Text Model Inference")

# Text input instead of file uploader
user_input = st.text_input("Enter your text", "")

if user_input:
    st.write("Your input:", user_input)
    
    # Process the input with the model
    output = model(user_input)
    
    # Display each sentence in the output list
    st.subheader("Model Output:")
    for i, sentence in enumerate(output, 1):
        st.write(f"{i}. {sentence}")
