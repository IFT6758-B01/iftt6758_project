import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("../dataset/simple_engineered/augmented_data.csv")

# Check for missing values in the 'distance_from_net' column
print(f"Number of missing values: {df['distance_from_net'].isna().sum()}")

# Option 1: Drop rows with missing values
df = df.dropna(subset=['distance_from_net'])

# OR Option 2: Impute missing values with the median
# df['distance_from_net'] = df['distance_from_net'].fillna(df['distance_from_net'].median())

# Use only the 'distance_from_net' feature and the target 'is_goal'
X = df[['distance_from_net']]  # Feature: distance
y = df['is_goal']              # Target: whether it is a goal (binary)

# Split the dataset into training and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model with default settings
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(X_val)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_val, y_pred)

# Display the accuracy
print(f"Validation Accuracy: {accuracy:.4f}")

# Optional: Analyze predictions
df_val = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred, 'Distance': X_val['distance_from_net']})
print(df_val.head(100))  # View a sample of predictions
