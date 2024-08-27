#this


from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

#defining the app
app = Flask(__name__)

# Load the model
model_path = 'my_model.keras' 
model = tf.keras.models.load_model(model_path)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the model prediction API. Use the /predict endpoint to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        if 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        features = np.array(data['features'])
        
        #check for feature shape
        if features.ndim != 3 or features.shape != (64, 64, 3):
            return jsonify({'error': 'Invalid feature shape'}), 400
        #the model shape
        features = features.reshape((1, *features.shape)) 

        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  
