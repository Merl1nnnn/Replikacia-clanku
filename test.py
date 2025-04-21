import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

model_file_size = 200
loading_str = f'_size_{model_file_size}'

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5, linecolor='gray')
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Bar charts for precision, recall, f1-score per class
def plot_classification_report(report_dict):
    metrics = ['precision', 'recall', 'f1-score']
    labels = list(report_dict.keys())[:-3]  # Skip 'accuracy', 'macro avg', 'weighted avg']

    x = np.arange(len(labels))
    width = 0.25

    # Values
    precision = [report_dict[label]['precision'] for label in labels]
    recall = [report_dict[label]['recall'] for label in labels]
    f1 = [report_dict[label]['f1-score'] for label in labels]

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ca02c')
    bars3 = ax.bar(x + width, f1, width, label='F1-score', color='#ff7f0e')

    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  # vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    # Add average lines
    for metric_values, color in zip([precision, recall, f1], ['#1f77b4', '#2ca02c', '#ff7f0e']):
        avg = np.mean(metric_values)
        ax.axhline(avg, linestyle='--', color=color, alpha=0.3, label=f'{color.capitalize()} Avg: {avg:.2f}')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics per Class', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()

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
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, list(class_map.keys()))

    report = classification_report(y_test, y_pred, target_names=class_map.keys(), output_dict=True)
    plot_classification_report(report)

if __name__ == "__main__":
    main()
