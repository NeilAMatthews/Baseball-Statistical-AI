import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os

MODEL_PATH = 'baseball_model.keras'
SCALER_PATH = 'scaler.pkl'

def create_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid') # Probability of a hit
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(data):
    # Assume last column is target 'is_hit'
    # and all other columns are features
    X = data.drop('is_hit', axis=1)
    y = data['is_hit']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = create_model(X_train.shape[1])
    print("Training model...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")
    
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

if __name__ == "__main__":
    # For testing purposes
    if os.path.exists('processed_data.csv'):
        df = pd.read_csv('processed_data.csv')
        train_model(df)
    else:
        print("processed_data.csv not found. Run data_loader.py first.")
