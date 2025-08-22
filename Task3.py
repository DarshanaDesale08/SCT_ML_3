import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog # For HOG features

# --- 1. Load and Preprocess Data ---
def load_images_and_labels(data_dir, img_size=(64, 64)):
    images = []
    labels = []
    for category in ['cats', 'dogs']:
        path = os.path.join(data_dir, category)
        class_num = 0 if category == 'cats' else 1 # 0 for cats, 1 for dogs
        for img_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, img_size)
                images.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    return np.array(images), np.array(labels)

# --- 2. Feature Extraction (Example with HOG) ---
def extract_hog_features(images):
    hog_features = []
    for img in images:
        features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False)
        hog_features.append(features)
    return np.array(hog_features)

# Main execution
data_directory = 'path/to/your/kaggle/cats_vs_dogs/dataset' # IMPORTANT: Change this path

# Load data
images, labels = load_images_and_labels(data_directory)

# Extract features
features = extract_hog_features(images) # Or flatten raw pixels: features = images.reshape(len(images), -1)

# --- 3. Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# --- 4. Train the SVM Model ---
svm_model = SVC(kernel='linear', C=1) # You can experiment with 'rbf' kernel and different C values
svm_model.fit(X_train, y_train)

# --- 5. Evaluate the Model ---
y_pred = svm_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
