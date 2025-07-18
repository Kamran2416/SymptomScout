import streamlit as st
from dotenv import load_dotenv
import os
import requests
import json
import pickle
import re
import concurrent.futures
import numpy as np
global_array = [0] * 40

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
api_url = os.getenv('GEMINI_API_URL')

st.set_page_config(
    page_title="SymptomScout",
    page_icon="🩺",
    layout="centered"
)

# Load models
def load_model(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as model_file:
            return pickle.load(model_file)
    else:
        st.error(f"Model file not found: {file_path}")
        st.stop()

nlp_model = load_model('./models/NLPModel.pkl')
vectorizer = load_model('./models/Vectorizer.pkl')
classifier_model = load_model('./models/DiseaseClassifier.pkl')

# Preprocess input text
def preprocess_input_text(input_text):
    processed_text = []
    for text in input_text:
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        processed_text.append(text)
    return processed_text

# Check if the input contains symptom-related terms
def contains_symptom_keywords(query):
    query = preprocess_input_text([query])
    symptom_keywords = [
        'skin problems', 'side pain', 'irritability', 'lower body pain', 'slow-healing', 'wounds', 'weight loss', 'blood in sputum', 'weakness', 'rapid heartbeat', 'hoarseness', 'neck pain', 'depression', 'skin rash', 'anxiety', 'skin irritation', 'high blood pressure', 'wheezing', 'retention of urine', 'immune system', 'coryza', 'cough', 'decreased appetite', 'shortness of breath', 'chest pain', 'blurred vision', 'numbness', 'irregular heartbeat', 'personality changes', 'abusing alcohol', 'diabetic', 'low libido', 'chest tightness', 'muscle weakness', 'erectile dysfunction', 'diarrhea', 'fatigue', 'muscle pain', 'lower abdominal pain', 'blurry vision', 'dizziness', 'abdominal pain', 'swollen lymph nodes', 'appetite changes', 'jaundice', 'swelling', 'poor coordination', 'problems with movement', 'insomnia', 'restlessness', 'painful periods', 'muscle tension', 'low back pain', 'burning abdominal pain', 'urinary problems', 'nasal congestion', 'nausea', 'leg cramps', 'hot flashes', 'ache all over', 'contagious', 'sweating', 'delayed growth', 'seizures', 'pregnancy', 'back pain', 'pelvic pain', 'body aches', 'memory loss', 'headaches', 'infertility', 'vomiting', 'skin lesions', 'weight gain', 'vaginal dryness', 'fainting', 'leg pain', 'allergic reaction', 'fever', 'thirst', 'sore throat', 'confusion', 'arm pain', 'skin swelling', 'menopause', 'heartburn', 'persistent cough', 'difficulty breathing', 'dark urine', 'joint pain', 'sharp chest pain', 'urination', 'stiff neck', 'hallucinations', 'mood swings', 'pain', 'difficulty swallowing', 'chills', 'rash', 'sharp abdominal pain', 'digestive problems', 'sexual problems', 'night sweats', 'headache', 'vision', 'hello', 'hair loss', 'difficulty concentrating', 'tremors', 'pain during intercourse', 'loss of appetite', 'tingling', 'hydrophobia', 'loss of sensation', 'disturbance of memory', 'healing', 'ache', 'arm', 'leg', 'aches'
    ]
    return any(keyword in query[0] for keyword in symptom_keywords)

# get response from the NLP model
def get_NLPModel_response(user_input):
    processed_text = preprocess_input_text([user_input])
    vectorized_text = vectorizer.transform(processed_text)
    predicted_label = nlp_model.predict(vectorized_text)
    return predicted_label[0]

def get_ClassificationModel_response(Symptom_array):
    input_data = np.array(Symptom_array).reshape(1, -1)
    y_pred = classifier_model.predict(input_data)
    y_proba = classifier_model.predict_proba(input_data)
    confidence = np.max(y_proba)

    if confidence > 0.98:
        return y_pred[0]
    else:
        return None

# Get the response from the Gemini API
def get_final_response(predicted_label):
    headers = {
        'Content-Type': 'application/json',
    }

    user_input = f"Predicted label: {predicted_label}. Give some info about this and suggest remedies, home remedies to cure this, and provide recommendations on managing it. Also give it"

    data = {
        "contents": [{
            "parts": [{"text": user_input}]
        }]
    }

    try:
        response = requests.post(f"{api_url}?key={api_key}", headers=headers, data=json.dumps(data))
        response.raise_for_status()
        api_response = response.json()

        if 'candidates' in api_response and len(api_response['candidates']) > 0:
            return api_response['candidates'][0]['content']['parts'][0]['text']
        else:
            return "I'm sorry, I didn't get a valid response."

    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting Gemini API: {e}")
        return "I'm sorry, there was an error processing your request."
    
def get_gemini_response(prompt):
    return "U Failed"

def handle_prompt(prompt):
    global global_array

    if contains_symptom_keywords(prompt):
        Symptom_array = get_NLPModel_response(prompt)  # Should return a list of 40 values (0s and 1s)

        # Update global_array by performing OR operation with Symptom_array
        global_array = [global_array[i] | Symptom_array[i] for i in range(len(global_array))]
        converted_array = [int(value) for value in global_array]
        print(converted_array)
        # Pass the updated global_array to the Classification model
        predicted_label = get_ClassificationModel_response(global_array)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if predicted_label:
                future_gemini_response = executor.submit(get_final_response, predicted_label)
                gemini_response = future_gemini_response.result()
            else:
                future_gemini_response = executor.submit(get_gemini_response, prompt)
                gemini_response = future_gemini_response.result()

        st.chat_message("SymptomScout").markdown(gemini_response)
        st.session_state.messages.append({'role': 'SymptomScout', 'content': gemini_response})
    else:
        st.chat_message("SymptomScout").markdown("Please enter your symptoms so I can assist you better. I am a medical-focused AI here to help!")
        st.session_state.messages.append({'role': 'SymptomScout', 'content': "Please enter your symptoms so I can assist you better. I am a medical-focused AI here to help!"})

st.title("Ask SymptomScout")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Ask SymptomScout")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    handle_prompt(prompt)