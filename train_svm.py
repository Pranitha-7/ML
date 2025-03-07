import pickle
from sklearn.svm import SVC
from keras.models import load_model, Model
import numpy as np
import os

# Load the CNN model (pretrained model with weights)
cnn_model = load_model('model/cnn_weights.hdf5')

# Extract the second-to-last layer (just before the fully connected layers) to use as features
cnn_features_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)

# Load the image dataset and labels
X = np.load('model/X.txt.npy')  # Loaded images array
Y = np.load('model/Y.txt.npy')  # Loaded labels array

# Normalize and preprocess the images (optional but recommended for neural networks)
X = X.astype('float32')
X = X / 255  # Normalize pixel values to range [0,1]

# Extract CNN features (from the CNN model, without the final output layer)
cnn_features = cnn_features_model.predict(X)

# Train the SVM model using the extracted CNN features
svm_cls = SVC(C=102.0, tol=1.9)  # Initialize SVM with specific hyperparameters
svm_cls.fit(cnn_features, Y)  # Train SVM model on extracted features and labels

# Save the trained SVM model to a file using pickle
with open('model/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_cls, f)  # Serialize and save the model

print("SVM model has been trained and saved as svm_model.pkl.")
