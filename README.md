# Price My Phone GH

**Price My Phone GH** is a Flask-based web application that predicts the price of a smartphone based on its model and generates an image of the phone using OpenAI's DALL-E model.

## Features

- Predicts the price of a smartphone based on its model.
- Displays similar phones based on the features of the input phone.
- Generates an image of the phone with a white background and its specifications.
- Provides autocomplete suggestions for phone names.

## Prerequisites

- Python 3.7 or higher
- Flask
- Joblib
- OpenAI API key

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/pricemyphonegh.git
    cd pricemyphonegh
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Set up your OpenAI API key:

    - Create a `.env` file in the root directory of the project.
    - Add your OpenAI API key to the `.env` file:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Start the Flask server:

    ```sh
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

## Application Structure

- `app.py`: The main Flask application file that handles routes, model loading, and predictions.
- `index.html`: The main HTML file that serves the web interface.
- `requirements.txt`: Contains all the dependencies required to run the application.

## `app.py` Overview

- **Imports and Configuration**:
    - Import necessary libraries and modules.
    - Set up logging configuration.
    - Define file paths for the model, phone features, label encoder, and features used during training.
    - Load the trained model, phone features, and label encoder.

- **Routes**:
    - `/autocomplete`: Provides autocomplete suggestions for phone names.
    - `/predict`: Predicts the price of the phone, finds similar phones, and generates an image of the phone using OpenAI's DALL-E model.
    - `/`: Serves the `home.html` file.
    - `/predict-page`: Serves the `index.html` file.

- **Helper Functions**:
    - `get_similar_phones`: Finds similar phones based on the features of the input phone.
    - `generate_image`: Generates an image of the phone using OpenAI's DALL-E model.

## Machine Learning Model

### Data Preparation

- **Data Loading**:
    - Load the data from a CSV file using `pandas`.
  
- **Data Cleaning and Imputation**:
    - Handle missing values by filling numerical columns with their median and categorical columns with their mode.

- **Feature Engineering**:
    - Combine `brand` and `model` into a single column `brand_model`.
    - Encode categorical features using `LabelEncoder`.

### Model Training and Evaluation

- **Correlation Analysis**:
    - Compute the correlation matrix to identify the most relevant features for predicting the price.

- **Feature Visualization**:
    - Visualize the distribution of key numerical features using histograms and boxplots.

- **Model Selection and Hyperparameter Tuning**:
    - Train and evaluate multiple models: Random Forest Regressor, Gradient Boosting Regressor, and Support Vector Regressor.
    - Use `GridSearchCV` for hyperparameter tuning.

- **Neural Network Model**:
    - Train a neural network using TensorFlow and Keras, with early stopping to prevent overfitting.

- **Stacking Regressor**:
    - Combine the predictions of multiple models using a stacking regressor to improve performance.

### Feature Importance

- **SHAP Values**:
    - Use SHAP (SHapley Additive exPlanations) to interpret the feature importance of the trained Random Forest model.

### Model Persistence

- **Save the Model and Features**:
    - Save the trained Gradient Boosting model and the features used during training using `joblib`.

## Example Usage in Jupyter Notebook

Here is an example of how the model prediction and image generation can be integrated in a Jupyter notebook:

```python
import requests
import json

# URL of the Flask server
url = 'http://127.0.0.1:5000/predict'

# Phone name to predict
phone_name = 'Apple_iPhone_11'

# Request payload
payload = {'phone_name': phone_name}

# Send the POST request
response = requests.post(url, json=payload)

# Get the response
data = response.json()

# Display the results
print(f"Predicted Price: {data['predicted_price']}")
print(f"Confidence Interval: {data['confidence_interval']}")
print(f"Similar Phones: {data['similar_phones']}")

# Display the generated image
from IPython.display import Image
Image(url=data['image_url'])
```

## Conclusion

This project demonstrates how to integrate machine learning models with web applications using Flask and how to utilize OpenAI's DALL-E model for image generation. The application provides a seamless user experience for predicting smartphone prices and visualizing the phone with its specifications.
