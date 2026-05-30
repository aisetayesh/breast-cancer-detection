# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
# --- Configuration ---
DATASET_PATH = "./dataset/cancer (1).csv"
TARGET_COLUMN = "diagnosis"
DROP_COLUMN = "id"
TEST_SIZE = 0.15
RANDOM_STATE = 42

# %%
# --- Load Data ---
try:
    data = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATASET_PATH}")
    exit() # Exit if the dataset file is not found

# %%
# --- Data Preprocessing ---
# Display first few rows and basic statistics
print("Dataset Head:")
print(data.head())
print("\nDataset Description:")
print(data.describe())

# Drop unnecessary columns (like 'id')
if DROP_COLUMN in data.columns:
    data = data.drop([DROP_COLUMN], axis=1)
else:
    print(f"Warning: Column '{DROP_COLUMN}' not found in dataset.")

# Convert target column to numerical (M=1 for malignant, B=0 for benign)
if TARGET_COLUMN in data.columns:
    data[TARGET_COLUMN] = data[TARGET_COLUMN].apply(lambda diagnosis: 1 if diagnosis == "M" else 0)
else:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in dataset.")
    exit()

# %%
# Separate features (X) and target (y)
y = data[TARGET_COLUMN]
x = data.drop([TARGET_COLUMN], axis=1)

# %%
# Normalize features (Min-Max scaling to range [0, 1])
# This helps algorithms that are sensitive to feature scales, like Logistic Regression.
x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
print("\nScaled Features (first 5 rows):")
print(x_scaled.head())

# %%
# --- Model Training ---
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Initialize and train Logistic Regression model
model = LogisticRegression(max_iter=2000) # Increased max_iter for convergence
model.fit(x_train, y_train)

# --- Model Evaluation ---
# Predict on the test set
y_pred = model.predict(x_test)

# %%
# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# %%
# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign (0)", "Malignant (1)"], yticklabels=["Benign (0)", "Malignant (1)"])
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("Actual Label", fontsize=12)
plt.show()

print("\nEvaluation complete.")


