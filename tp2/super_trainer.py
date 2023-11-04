import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

#------------------------------------------

import huFile
import labels
import variables_toni

#------------------------------------------

def predict_label(classifier, image_path):
    hu_moments = huFile.calculate_hu_moments(image_path)
    predicted_label = classifier.predict([hu_moments])[0]
    predicted_label_str = labels.int_to_label(predicted_label)
    return predicted_label, predicted_label_str

#------------------------------------------

# Load the first CSV file
csv_file1 = variables_toni.extra_huMoments_path
hu_moments_df1 = pd.read_csv(csv_file1)

# Load the second CSV file
csv_file2 = variables_toni.huMoments_path
hu_moments_df2 = pd.read_csv(csv_file2)

# Concatenate the data from both files
hu_moments_df = pd.concat([hu_moments_df1, hu_moments_df2], ignore_index=True)

#------------------------------------------

# Split the data into features (X) and labels (y)
X = hu_moments_df['hu_moments'].apply(eval).tolist()
y = hu_moments_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#------------------------------------------

# 2. k-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"k-Nearest Neighbors Accuracy: {accuracy_knn}")

# Save the K-NN model
joblib.dump(knn_classifier, 'knn_super_model.joblib')

# Example: Predict a new shape using KNN
new_shape_path_knn = variables_toni.triangle6
predicted_label_knn, predicted_label_str_knn = predict_label(knn_classifier, new_shape_path_knn)
print(f"Predicted Label for the new shape (KNN): {predicted_label_knn} ({predicted_label_str_knn})")