from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd
import numpy as np

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X = train_df.iloc[:, 1:-1]
y = train_df["y"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost model with early stopping
model = CatBoostRegressor(iterations=300, learning_rate=0.1, depth=6, early_stopping_rounds=50, verbose=0)

# Train model with tqdm progress bar
for i in tqdm(range(1, 301)):
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)

# Predict on test data
X_test = test_df.iloc[:, 1:]
y_pred = model.predict(X_test)

# Identify top 33% of predicted values
threshold = np.percentile(y_pred, 67)
top_33_percent_mask = y_pred >= threshold

# Create submission file
submission_df = pd.read_csv('sample_submission.csv')
submission_df['y'] = y_pred
submission_df.to_csv('cat.csv', index=False)

print(f"Top 33% threshold: {threshold:.4f}")
print(f"Number of samples in top 33%: {sum(top_33_percent_mask)}")
