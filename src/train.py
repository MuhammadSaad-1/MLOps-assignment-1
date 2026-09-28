import os
import yaml
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def main():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    p = params['train']
    
    train_data = np.load('data/processed/train.npz')
    val_data = np.load('data/processed/val.npz')
    
    x_train, y_train = train_data['x'], train_data['y']
    x_val, y_val = val_data['x'], val_data['y']
    
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(p['dense_units'], activation='relu'),
        Dropout(p['dropout_rate']),
        Dense(10, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=p['learning_rate'])
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, 
                        validation_data=(x_val, y_val),
                        epochs=p['epochs'], 
                        batch_size=p['batch_size'])
    
    os.makedirs('models', exist_ok=True)
    model.save('models/model.h5')
    pd.DataFrame(history.history).to_csv('models/history.csv', index=False)

if __name__ == '__main__':
    main()
