import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

# --- 1. Title and Description
st.title("Flexible Chatbot")
st.write("Choose a model and start chatting!")


# --- 2. Define Available Models
AVAILABLE_MODELS = {
    "OPT-30B": {"model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "model_id": "facebook/opt-30b"},
    "FLAN-T5-XXL": {"model_class": AutoModelForSeq2SeqLM, "tokenizer_class": AutoTokenizer, "model_id": "google/flan-t5-xxl"},
    "Dialo-medium": {"model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "model_id": "microsoft/Dialo-medium"},
    "GPT2": {"model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "model_id":"gpt2"},
}


# --- 3. Load Model and Tokenizer (using a function to handle different model types)
@st.cache_resource
def load_model_and_tokenizer(model_name):
    model_info = AVAILABLE_MODELS[model_name]
    tokenizer = model_info["tokenizer_class"].from_pretrained(model_info["model_id"], use_fast = False)
    model = model_info["model_class"].from_pretrained(model_info["model_id"])
    return tokenizer, model


# --- 4. Initialize Chat History (using session_state)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_model" not in st.session_state:
   st.session_state.selected_model = "OPT-30B"


# --- 5. UI Elements
selected_model = st.selectbox("Select a Model:", options=list(AVAILABLE_MODELS.keys()), index = list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model) )
user_input = st.text_input("Your message:", key="input") #key used to avoid a problem with multiple inputs and the same name
submit_button = st.button("Send")
clear_button = st.button("Clear Chat")

max_length = st.slider("Max Length:", min_value=50, max_value=500, value=200, step=10)
temperature = st.slider("Temperature:", min_value=0.1, max_value=1.5, value=1.0, step=0.1)
top_p_enabled = st.checkbox("Enable Top P Sampling", value = False)
if top_p_enabled:
   top_p = st.slider("Top P:", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
else:
   top_p = 0.0


# --- 6. Chat Logic
if submit_button and user_input:
    try:
        # Load selected model and tokenizer
        if selected_model != st.session_state.selected_model:
             tokenizer, model = load_model_and_tokenizer(selected_model)
             st.session_state.selected_model = selected_model
        else:
             tokenizer, model = load_model_and_tokenizer(selected_model)



        # Append user message
        st.session_state.chat_history.append(("You", user_input))

        # Create a prompt using previous messages and the current user input
        prompt = "".join([f"{sender}: {message} " for sender, message in st.session_state.chat_history])


        # Process input for the model
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate chatbot response using sampling parameters
        if top_p_enabled:
            output_ids = model.generate(
                 input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample = True,
             )
        else:
            output_ids = model.generate(
                 input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample = True,
            )
        bot_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Append bot message to the history
        st.session_state.chat_history.append(("Bot", bot_response))

        # Clear user input
        st.session_state.input = ""

    except Exception as e:
        st.error(f"An error occurred: {e}")


# -- 7 Clear Chat History
if clear_button:
        st.session_state.chat_history = []



# --- 8. Display Chat History
for sender, message in st.session_state.chat_history:
    st.text(f"{sender}: {message}")