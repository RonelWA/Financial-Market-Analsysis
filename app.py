from flask import Flask, request, jsonify, send_from_directory
import openai
import pickle
import numpy as np
import joblib
import os
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__, static_folder='')

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler('app.log')])

logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define file paths
model_path = 'gradient_boosting_model.pkl'
features_path = 'phone_features.pkl'
encoder_path = 'label_encoder.pkl'
features_used_path = 'features_used_for_training.pkl'

# Check if files exist
if not os.path.exists(model_path):
    logger.error(f"Model file not found: {model_path}")
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(features_path):
    logger.error(f"Phone features file not found: {features_path}")
    raise FileNotFoundError(f"Phone features file not found: {features_path}")
if not os.path.exists(encoder_path):
    logger.error(f"Label encoder file not found: {encoder_path}")
    raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
if not os.path.exists(features_used_path):
    logger.error(f"Features used file not found: {features_used_path}")
    raise FileNotFoundError(f"Features used file not found: {features_used_path}")

# Load the trained model
model = joblib.load(model_path)
logger.info("Model loaded successfully.")

# Load the phone features
phone_features = joblib.load(features_path)

# Load the label encoder
with open(encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

# Load the features used during training
features_used_for_training = joblib.load(features_used_path)
logger.debug(f"Features used for training: {features_used_for_training}")

# Log the columns of the DataFrame to inspect (temporary code for debugging)
logger.debug(f"Phone Features Columns: {phone_features.columns}")

# Normalize the phone features for distance calculation
scaler = StandardScaler()
scaled_phone_features = scaler.fit_transform(phone_features[features_used_for_training])



def get_similar_phones(features, top_n=5):
    try:
        logger.debug("Starting get_similar_phones function")

        # Ensure required columns are present
        if 'ram(GB)' not in phone_features.columns or 'storage(GB)' not in phone_features.columns:
            logger.error("Required columns missing in phone_features DataFrame")
            return []

        # Combine the features for distance calculation
        current_phone_features = features[['ram(GB)', 'storage(GB)']].values
        logger.debug(f"Current phone features: {current_phone_features}")

        # Extract features for all phones
        all_phones_features = phone_features[['ram(GB)', 'storage(GB)']].values
        logger.debug(f"All phones features: {all_phones_features[:5]}")  # Log only the first 5 for brevity

        # Calculate the Euclidean distances
        distances = euclidean_distances(current_phone_features, all_phones_features).flatten()
        logger.debug(f"Calculated distances: {distances[:5]}")  # Log only the first 5 for brevity

        # Create a DataFrame with distances
        phone_features_copy = phone_features.copy()
        phone_features_copy['distance'] = distances

        # Exclude the current phone itself
        similar_phones = phone_features_copy[phone_features_copy['brand_model'] != features['brand_model'].values[0]].sort_values('distance')
        logger.debug(f"Similar phones after sorting: {similar_phones[['brand_model', 'distance']].head(top_n)}")

        # Select the top N similar phones
        similar_phones = similar_phones.head(top_n)['brand_model'].values

        similar_phones_decoded = label_encoder.inverse_transform(similar_phones.astype(int))
        logger.debug(f"Similar phones (decoded): {similar_phones_decoded}")

        return similar_phones_decoded.tolist()
    except Exception as e:
        logger.error(f"Error in get_similar_phones: {str(e)}")
        return []

def generate_image(phone_name):
    try:
        prompt = (
            f"Create an image of the {phone_name} smartphone on a white background and show the phone's specifications (screen size, camera resolution, RAM, and storage capacity) beside it in a clear layout. Ensure the phone looks sleek and modern"
        )
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
        return image_url
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return None

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '').lower().replace(' ', '_')
    suggestions = [label for label in label_encoder.classes_ if query in label.lower()]
    return jsonify({'suggestions': suggestions})
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        phone_name = data.get('phone_name')
        
        if not phone_name:
            logger.warning("No phone name provided in the request")
            return jsonify({'error': 'Phone name is required'}), 400
        
        phone_name = phone_name.strip().replace(' ', '_').lower()
        
        encoded_brand_model = None
        for label in label_encoder.classes_:
            normalized_label = label.lower().replace(' ', '_')
            if normalized_label == phone_name:
                encoded_brand_model = label_encoder.transform([label])[0]
                break
        
        if encoded_brand_model is None:
            logger.warning(f"Phone not found: {phone_name}")
            return jsonify({'error': f'Phone not found: {phone_name}'}), 404

        features = phone_features[phone_features['brand_model'] == encoded_brand_model]
        if features.empty:
            logger.warning(f"Features not found for phone: {phone_name}")
            return jsonify({'error': 'Features not found for phone'}), 404

        features_with_brand_model = features.copy()
        features_with_brand_model['brand_model'] = encoded_brand_model

        features_for_prediction = features_with_brand_model[features_used_for_training]
        logger.debug(f"Prepared Features for Prediction: {features_for_prediction.columns}")

        price = model.predict(features_for_prediction)[0]
        confidence_interval = [price - 100, price + 100]
        similar_phones = get_similar_phones(features_with_brand_model)

        formatted_price = f"GH₵ {price:.2f}"
        formatted_confidence_interval = [f"GH₵ {ci:.2f}" for ci in confidence_interval]

        storage = str(features_with_brand_model['storage(GB)'].values[0])
        ram = str(features_with_brand_model['ram(GB)'].values[0])

        image_url = generate_image(phone_name)

        response = {
            'predicted_price': formatted_price,
            'confidence_interval': formatted_confidence_interval,
            'similar_phones': similar_phones,
            'image_url': image_url,
            'specifications': {
                'storage': storage,
                'ram': ram
            }
        }
        logger.info(f"Prediction successful for {phone_name}: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred during prediction'}), 500
@app.route('/')
def home():
    return send_from_directory('', 'home.html')

@app.route('/predict-page')
def predict_page():
    return send_from_directory('', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
