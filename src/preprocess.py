import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    test_size = params['preprocess']['test_size']
    seed = params['preprocess']['seed']
    
    train_data = np.load('data/raw/train.npz')
    test_data = np.load('data/raw/test.npz')
    
    x_train_full = train_data['x'].astype('float32') / 255.0
    y_train_full = train_data['y']
    
    x_test = test_data['x'].astype('float32') / 255.0
    y_test = test_data['y']
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=test_size, random_state=seed
    )
    
    os.makedirs('data/processed', exist_ok=True)
    np.savez_compressed('data/processed/train.npz', x=x_train, y=y_train)
    np.savez_compressed('data/processed/val.npz', x=x_val, y=y_val)
    np.savez_compressed('data/processed/test.npz', x=x_test, y=y_test)

if __name__ == '__main__':
    main()
