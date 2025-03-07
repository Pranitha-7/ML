import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image, ImageTk
import seaborn as sns
import tkinter  as tk
from tkinter import filedialog
from keras.models import load_model, Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

root=None

# Load pre-trained CNN model
cnn_model = load_model('model/cnn_weights.hdf5')

# Extract CNN feature extractor
cnn_feature_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Class labels (0 = Body, 1 = Hand, 2 = Leg, 3 = None)
class_labels = ['Body', 'Hand', 'Leg', 'None']

# Function to preprocess image for CNN input
def preprocess_image(img):
    img = cv2.resize(img, (32, 32))  # Resize to match CNN input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prepare dataset
X = []
y = []

dataset_path = "dataset/"  # Modify this path
image_filenames = []  # Store image file names

for label, category in enumerate(class_labels):
    category_path = os.path.join(dataset_path, category)
    for filename in os.listdir(category_path):
        img_path = os.path.join(category_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        features = cnn_feature_model.predict(preprocess_image(img))
        X.append(features.flatten())  # Flatten feature vector
        y.append(label)
        image_filenames.append(filename)  # Store image filename

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(probability=True)  # Enable probability estimation
svm_model.fit(X_train, y_train)

# Save trained SVM model
with open('model/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

print("✅ Training Completed!")
# Preprocess the image for CNN
# def preprocess_image(img):
#     try:
#         img = cv2.resize(img, (32, 32))  # Resize to CNN input size
#         img = img / 255.0  # Normalize pixel values
#         img = np.expand_dims(img, axis=0)  # Reshape for CNN model input
#         return img
#     except Exception as e:
#         raise ValueError(f"Error during image preprocessing: {e}")

# Predict the class of the image using SVM and CNN

import os
import shutil

# 🟢 Generate predictions before using them
predictions = svm_model.predict(X_test)  
# predictions = predictions.argmax(axis=1)  # Convert softmax output to class indices

misclassified_dir = "misclassified_images/"
os.makedirs(misclassified_dir, exist_ok=True)  # Ensure the directory exists

for i in range(len(y_test)):
    if predictions[i] != y_test[i]:  # If prediction is incorrect
        actual_label_index = y_test[i]  # Correct label index
        predicted_label_index = predictions[i]  # Predicted label index

        # Ensure labels are valid
        if 0 <= actual_label_index < len(class_labels) and 0 <= predicted_label_index < len(class_labels):
            actual_label = class_labels[actual_label_index]  # Get correct class name
            predicted_label = class_labels[predicted_label_index]  # Get predicted class name
        else:
            print(f"⚠️ Invalid label index: {actual_label_index} or {predicted_label_index}")
            continue

        # Retrieve correct filename
        original_filename = image_filenames[i]

        # Construct the correct original path
        original_path = os.path.join(dataset_path, actual_label, original_filename)

        # Construct new path with predicted label
        new_path = os.path.join(misclassified_dir, f"wrong_{i}_pred_{predicted_label}_{original_filename}")

        # Copy misclassified image only if it exists
        if os.path.exists(original_path):
            shutil.copy(original_path, new_path)
            print(f"✅ Copied misclassified image: {original_filename}")
        else:
            print(f"⚠️ File not found: {original_path}, skipping...")


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2
)


from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly']
}

grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, verbose=2)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)


def predict_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Invalid image file.")

        preprocessed_img = preprocess_image(img)
        cnn_features = cnn_feature_model.predict(preprocessed_img)

        # Get prediction probabilities from SVM
        probabilities = svm_model.predict_proba(cnn_features)[0]
        predicted_class = np.argmax(probabilities)  # Class with highest probability
        max_confidence = np.max(probabilities)  # Highest confidence score

        # Set confidence threshold
        threshold = 0.85  # Adjust based on performance

        # If confidence is too low, return "None"
        if max_confidence < threshold:
            return "None"

        predicted_label = class_labels[predicted_class]
        return predicted_label

    except Exception as e:
        return f"Error: {e}"

# Browse and classify an image
def browse_image():
    try:
        filename = filedialog.askopenfilename(
            initialdir="/",
            title="Select an Image",
            filetypes=(("Image Files", "*.jpg *.png"),)
        )
        if filename:
            prediction = predict_image(filename)
            # result_label.config(text=f"Prediction: {prediction}", fg="#FF1D58",)
            
            # Create a new window for displaying the image and result
            new_window = tk.Toplevel(root)  # Creates a new window
            new_window.title("Image and Prediction")
            new_window.geometry("10000x10000")  # Adjust the size of the new window

            # Create a frame for the image and prediction
            image_frame = tk.Frame(new_window ,bg="#E1F8DC")
            image_frame.pack(fill="both", expand=True)

            # Display the image in the new window
            img = Image.open(filename)
            img = img.resize((1200, 600))  # Resize image to fit the new window
            img_tk = ImageTk.PhotoImage(img)

            image_label = tk.Label(image_frame, image=img_tk)
            image_label.image = img_tk  # Keep a reference to the image
            image_label.pack(pady=10)

            # Display the prediction in the new window
            prediction_label = tk.Label(image_frame, text=f"Victim Part Identified: {prediction}", font=("Arial", 35,"bold"), fg="Blue" )
            prediction_label.pack(pady=10)

    except Exception as e:
        result_label.config(text=f"Error: {e}", fg="red")


# Display the selected image in the GUI
def display_image(image_path, image_label):
    try:
        img = Image.open(image_path)
        img_tk = ImageTk.PhotoImage(img)  # Convert image to a format tkinter can use
        image_label.config(image=img_tk)  # Update the label with the image
        image_label.image = img_tk  # Keep a reference to the image
    except Exception as e:
        print(f"Error displaying image: {e}") 

# def load_dataset():
#     global X, Y
#     if os.path.exists("model/X.txt.npy"):
#         X = np.load("model/X.txt.npy")
#         Y = np.load("model/Y.txt.npy")
#     else:
#         for root, dirs, directory in os.walk("dataset"):  # Update with your dataset path
#             for j in range(len(directory)):
#                 name = os.path.basename(root)
#                 if "Thumbs.db" not in directory[j]:
#                     img = cv2.imread(root + "/" + directory[j])
#                     img = cv2.resize(img, (32, 32))
#                     X.append(img)
#                     label = labels.index(name)  # Replace with proper label logic
#                     Y.append(label)
#         X = np.asarray(X)
#         Y = np.asarray(Y)
#         np.save("model/X.txt", X)
#         np.save("model/Y.txt", Y)
#     result_label.config(text=f"Dataset loaded: {len(X)} images, {len(labels)} classes")


# def show_dataset_graph():
#     names, count = np.unique(Y, return_counts=True)
#     plt.figure(figsize=(6, 3))
#     plt.bar(names, count, color="green")
#     plt.xticks(names, labels)
#     plt.xlabel("Class Labels")
#     plt.ylabel("Count")
#     plt.title("Dataset Class Label Distribution")
#     plt.show()

# def train_cnn():
#     global cnn_model
#     X_normalized = X.astype("float32") / 255
#     Y_categorical = to_categorical(Y)
#     X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y_categorical, test_size=0.2)
#     cnn_model = Sequential()
#     cnn_model.add(Convolution2D(32, (3, 3), input_shape=(32, 32, 3), activation="relu"))
#     cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
#     cnn_model.add(Convolution2D(32, (3, 3), activation="relu"))
#     cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
#     cnn_model.add(Flatten())
#     cnn_model.add(Dense(units=256, activation="relu"))
#     cnn_model.add(Dense(units=len(labels), activation="softmax"))
#     cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#     if not os.path.exists("model/cnn_weights.hdf5"):
#         checkpoint = ModelCheckpoint("model/cnn_weights.hdf5", save_best_only=True, verbose=1)
#         cnn_model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_test, y_test), callbacks=[checkpoint])
#     else:
#         cnn_model.load_weights("model/cnn_weights.hdf5")
#     result_label.config(text="CNN model trained successfully")





# def train_svm():
#     global svm_model
#     features = cnn_model.predict(X)
#     X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.2)
#     svm_model = SVC(C=102.0, tol=1.9)
#     svm_model.fit(X_train, y_train)
#     predict = svm_model.predict(X_test)
#     calculate_metrics("SVM", predict, y_test)


# def train_dt():
#     global dt_cls
#     features = cnn_model.predict(X)
#     X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.2)
#     dt_cls = DecisionTreeClassifier()
#     dt_cls.fit(X_train, y_train)
#     predict = dt_cls.predict(X_test)
#     calculate_metrics("Decision Tree", predict, y_test)

# def train_rf():
#     global rf_cls
#     features = cnn_model.predict(X)
#     X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.2)
#     rf_cls = RandomForestClassifier(n_estimators=5)
#     rf_cls.fit(X_train, y_train)
#     predict = rf_cls.predict(X_test)
#     calculate_metrics("Random Forest", predict, y_test)



# def calculate_metrics(algorithm, predict, y_test):
#     acc = accuracy_score(y_test, predict) * 100
#     prec = precision_score(y_test, predict, average="macro") * 100
#     rec = recall_score(y_test, predict, average="macro") * 100
#     f1 = f1_score(y_test, predict, average="macro") * 100
#     conf_matrix = confusion_matrix(y_test, predict)
#     plt.figure(figsize=(6, 3))
#     sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels= labels, yticklabels= labels)
#     plt.title(f"{algorithm} Confusion Matrix")
#     plt.ylabel("True Label")
#     plt.xlabel("Predicted Label")
#     plt.show()
#     result_label.config(text=f"{algorithm} - Accuracy: {acc:.2f}%, Precision: {prec:.2f}%, Recall: {rec:.2f}%, F1: {f1:.2f}%")




# Build the GUI
# Create the Canvas and Scrollbar
def create_gui():
    root = tk.Tk()
    root.title("Image Classifier")
    root.geometry("600x700")
    root.configure(bg="#f4f4f4")

    # Title Label
    title_label = tk.Label(
        root,
        text="DISASTER VICTIM DETECTION",
        font=("Helvetica", 22, "bold"),
        bg="#4CAF50",
        fg="white",
        padx=20,
        pady=10,
    )
    title_label.pack(fill="x", pady=10)

    # Load Dataset Button
    # load_dataset_button = tk.Button(
    #     root,
    #     text="Load Dataset",
    #     command=load_dataset,
    #     font=("Arial", 12),
    #     bg="#28a745",
    #     fg="white",
    #     activebackground="#218838",
    #     activeforeground="white",
    #     relief="raised",
    #     width=20,
    #     height=2,
    # )

    # Show Dataset Graph Button
    # show_dataset_graph_button = tk.Button(
    #     root,
    #     text="Show Dataset Graph",
    #     command=show_dataset_graph,
    #     font=("Arial", 12),
    #     bg="#FF5733",
    #     fg="white",
    #     activebackground="#C70039",
    #     activeforeground="white",
    #     relief="raised",
    #     width=20,
    #     height=2,
    # )

    # Train CNN Button
    # train_cnn_button = tk.Button(
    #     root,
    #     text="Train CNN",
    #     command=train_cnn,
    #     font=("Arial", 12),
    #     bg="#28a745",
    #     fg="white",
    #     activebackground="#218838",
    #     activeforeground="white",
    #     relief="raised",
    #     width=20,
    #     height=2,
    # )

    # Train SVM Button
    # train_svm_button = tk.Button(
    #     root,
    #     text="Train SVM",
    #     command=train_svm,
    #     font=("Arial", 12),
    #     bg="#007BFF",
    #     fg="white",
    #     activebackground="#0056b3",
    #     activeforeground="white",
    #     relief="raised",
    #     width=20,
    #     height=2,
    # )

    # Train Random Forest Button
    # train_rf_button = tk.Button(
    #     root,
    #     text="Train Random Forest",
    #     command=train_rf,
    #     font=("Arial", 12),
    #     bg="#FF5733",
    #     fg="white",
    #     activebackground="#C70039",
    #     activeforeground="white",
    #     relief="raised",
    #     width=20,
    #     height=2,
    # )

    # Train Decision Tree Button
    # train_dt_button = tk.Button(
    #     root,
    #     text="Train Decision Tree",
    #     command=train_dt,
    #     font=("Arial", 12),
    #     bg="#FFC300",
    #     fg="white",
    #     activebackground="#FF5733",
    #     activeforeground="white",
    #     relief="raised",
    #     width=20,
    #     height=2,
    # )

    # Browse Button
    browse_button = tk.Button(
        root,
        text="Browse Image",
        command=browse_image,
        font=("Arial", 16, "bold"),  # Increase font size
        bg="#845BB3",  # Button background color
        fg="white",  # Text color
        activebackground="#0056b3",  # Active background color
        activeforeground="white",  # Active text color
        relief="raised",
        width=40,  # Button width
        height=2,  # Button height
    )

    # Result Label
    global result_label
    result_label = tk.Label(
        root,
        text="",
        font=("Arial", 50, "bold"),
        bg="#f4f4f4",
        fg="green",
        wraplength=400,
        justify="center",
        
    )
    result_label.pack(pady=10)

    # Frame to hold buttons horizontally
    button_frame = tk.Frame(root)
    button_frame.pack(side="bottom", fill="x", pady=20)

    # Add buttons to the frame (packed horizontally)
    #load_dataset_button.pack(side="left", padx=10)
    #show_dataset_graph_button.pack(side="left", padx=10)
    #train_cnn_button.pack(side="left", padx=10)
    #train_svm_button.pack(side="left", padx=10)
    #train_rf_button.pack(side="left", padx=10)
    #train_dt_button.pack(side="left", padx=10)
    browse_button.place(relx=0.5, rely=0.5, anchor="center")

    
   

    # Run the Tkinter main loop
    root.mainloop()

# Run the application
# 
if __name__ == "__main__":
    create_gui()
