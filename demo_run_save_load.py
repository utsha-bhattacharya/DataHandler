from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from model_save_load import save_model, load_model

# Load sample data
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model to disk
save_path = "random_forest_model.pkl"
save_model(model, save_path)
print(f"Model saved to {save_path}")

# Load the model from disk
loaded_model = load_model(save_path)
print("Model loaded from disk.")

# Use loaded model to predict
y_pred = loaded_model.predict(X_test)
print("Predictions:", y_pred)
