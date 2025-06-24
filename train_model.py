import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ✅ 1. Dataset path
data_dir = r'C:\Users\harsh\OneDrive\Desktop\java script\matploitlib\matplotlib\moodmate\archive\train'  # ⚠️ Make sure this path matches your folder structure

# ✅ 2. Emotion categories
categories = os.listdir(data_dir)
print("✅ Categories found:", categories)

X, y = [], []

# ✅ 3. Load and preprocess images
for category in categories:
    category_path = os.path.join(data_dir, category)
    label = categories.index(category)

    for img_name in os.listdir(category_path):
        try:
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            X.append(img.flatten())
            y.append(label)
        except:
            pass  # Skip unreadable images

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} images successfully.")

# ✅ 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 5. Define pipeline (Scaler + PCA + Logistic Regression)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('clf', LogisticRegression(max_iter=1000)) 
])

# ✅ 6. Grid Search Hyperparameters (using lbfgs only)
param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__solver': ['lbfgs']  # ✅ liblinear removed to avoid warning
}

# ✅ 7. Grid search and training
grid = GridSearchCV(pipeline, param_grid, cv=3, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# ✅ 8. Save best model
joblib.dump(grid.best_estimator_, 'mood_model.pkl')
print("\n✅ Model saved as 'mood_model.pkl'")

# ✅ 9. Evaluate performance
y_pred = grid.predict(X_test)
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
