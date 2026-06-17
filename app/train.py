from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os


def train_model():
    # Load built-in Wine dataset from sklearn
    data = load_wine()

    X = data.data
    y = data.target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Test model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model trained successfully")
    print(f"Accuracy: {accuracy:.4f}")

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    train_model()