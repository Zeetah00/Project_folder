This repository contains the complete project setup used for training the CNN model used in my project.

Project Structure
The project directory contains the following files:
- data_preprocessing.py: A Python script for preprocessing the CSV dataset files. It handles tasks such as loading, cleaning, and normalizing the data, and preparing it for model training.
- train_cnn.py: A Python script to define, train, and save a the CNN model using the preprocessed data.
- app.py: The Flask application script that loads the trained model and exposes an API endpoint (/predict) for making predictions with the model.
- requirements.txt: A text file listing all Python dependencies required for the project. This file is used for installing the necessary packages.
- Dockerfile: A Dockerfile to create a Docker image for running the Flask application. It includes instructions for setting up the environment, installing dependencies, and running the application.
- my_model.keras: The saved CNN model in Keras format. This file is loaded by the Flask application to make predictions.
- then all the CSV dataset files used to train the model. These datasets are necessary for running the data_preprocessing.py and train_cnn.py scripts.
