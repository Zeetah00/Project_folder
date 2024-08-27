import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from data_preprocessing import preprocess_data

#all file datasets load
file_paths = [
    'C:/Users/USER/Project_train/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX',
    'C:/Users/USER/Project_train/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX',
    'C:/Users/USER/Project_train/Friday-WorkingHours-Morning.pcap_ISCX',
    'C:/Users/USER/Project_train/Monday-WorkingHours.pcap_ISCX',
    'C:/Users/USER/Project_train/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX',
    'C:/Users/USER/Project_train/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX',
    'C:/Users/USER/Project_train/Tuesday-WorkingHours.pcap_ISCX',
    'C:/Users/USER/Project_train/Wednesday-workingHours.pcap_ISCX'
]


#call the function in the preprocess file
X, y = preprocess_data(file_paths)

#shape y to classify
y = to_categorical(y)

#then split the datasets into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#defining  the model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')  # Output layer: number of classes
])

# then compiling the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])


history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

# Save the model
model.save('my_model.keras')
