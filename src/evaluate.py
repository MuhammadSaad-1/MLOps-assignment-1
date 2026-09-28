import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

def main():
    test_data = np.load('data/processed/test.npz')
    x_test, y_test = test_data['x'], test_data['y']
    
    model = load_model('models/model.h5')
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    
    metrics = {
        'test_loss': float(loss),
        'test_accuracy': float(accuracy)
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main()
