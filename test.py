import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

model_file_size = 200
loading_str = f'_size_{model_file_size}'

def main():
    # Loading test features
    try:
        vgg_features_test = np.load(f"features{loading_str}/vgg_test_features.npy")  # Updated file name
        xception_features_test = np.load(f"features{loading_str}/xcep_test_features.npy")  # Updated file name
    except FileNotFoundError as e:
        print(f"Error loading features: {e}")
        return

    # Combining features
    X_test = np.concatenate([vgg_features_test, xception_features_test], axis=1)

    # Loading target labels
    try:
        y_labels = np.load(f"features{loading_str}/y.npy")  # Updated file name
        _, y_test = train_test_split(y_labels, test_size=0.2, stratify=y_labels, random_state=42)
    except FileNotFoundError as e:
        print(f"Error loading labels: {e}")
        return

    # Loading class map
    try:
        class_map = np.load(f"features{loading_str}/class_map.npy", allow_pickle=True).item()  # Updated file name
    except ValueError as e:
        print(f"Error loading class map: {e}")
        return

    id_to_class = {v: k for k, v in class_map.items()}

    # Loading trained model
    try:
        xgb_model = joblib.load(f"xgb_model{loading_str}.pkl")  # Updated file name
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return

    # Prediction
    y_pred = xgb_model.predict(X_test)

    # Report
    print("🔎 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=[id_to_class[i] for i in sorted(id_to_class)]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
