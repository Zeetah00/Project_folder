#uses the python 3.9 image as the base
FROM python:3.9-slim

#working directory in the container
WORKDIR /app

#install pip latest version
RUN pip install --upgrade pip

#Install TensorFlow and TensorFlow IO and made sure it match the tensorflow in the model
RUN pip install tensorflow==2.17.0 tensorflow-io

#send the requirements file into the container
COPY requirements.txt .

#this installs all in the requirements file
RUN pip install --no-cache-dir -r requirements.txt

#the application code comes into the container
COPY . .

# Make port 5000 available 
EXPOSE 5000

ENV FLASK_APP=app.py


CMD ["flask", "run", "--host=0.0.0.0"]
