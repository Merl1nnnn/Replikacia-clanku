import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import VGG16, Xception
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xcep_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from xgboost import XGBClassifier
import joblib
import logging
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# === ЛОГИРОВАНИЕ ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === НАСТРОЙКА GPU ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info("✅ GPU доступен: %s", gpus)
else:
    logger.warning("⚠️ GPU не найден. Используется CPU.")

# === ПАРАМЕТРЫ ===
IMG_SIZE = 200
DATASET_PATH = "grape_leaf_dataset"
PART_OF_DATASET = 0.3  # используй только часть данных, чтобы не упасть по памяти
SEED = 42
np.random.seed(SEED)

# === ФУНКЦИИ ===

def remove_background(image):
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(img_cv, img_cv, mask=mask)
    result[mask == 0] = [255, 255, 255]
    return result

def preprocess_image(path):
    try:
        image = Image.open(path).convert("RGB")
        img = remove_background(image)
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return resized / 255.0
    except Exception as e:
        logger.error("Ошибка в %s: %s", path, e)
        return np.zeros((IMG_SIZE, IMG_SIZE, 3))
    
def augment_image(image, class_idx):
    if class_idx in [0, 1]:  # Для классов 0 и 1 применяем сглаженную аугментацию
        datagen = ImageDataGenerator(
            rotation_range=10,  # Уменьшенный диапазон вращения
            width_shift_range=0.05,  # Уменьшенный диапазон горизонтального сдвига
            height_shift_range=0.05,  # Уменьшенный диапазон вертикального сдвига
            zoom_range=0.05,  # Уменьшенный диапазон масштабирования
            horizontal_flip=True  # Оставляем только горизонтальное отражение
        )
    else:  # Для остальных классов стандартная аугментация
        datagen = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
    
    image = np.expand_dims(image, 0)
    return [datagen.flow(image, batch_size=1).__next__()[0]]

def load_dataset(path, max_per_class=750, max_per_difficult_class=1500):
    X, y = [], []
    class_map = {name: idx for idx, name in enumerate(sorted(os.listdir(path)))}
    logger.info("Классы: %s", class_map)
    
    for label, idx in class_map.items():
        folder = os.path.join(path, label)
        images = os.listdir(folder)
        np.random.shuffle(images)
        
        # Устанавливаем лимит на количество изображений в зависимости от класса
        if idx in [0, 1]:  # Для классов 0 и 1
            images = images[:max_per_difficult_class]
        elif idx in [2, 3]:  # Для классов 2 и 3
            images = images[:max_per_class]
        else:
            logger.warning(f"Класс {idx} не обработан, пропускаем...")
            continue

        for file in images:
            img_path = os.path.join(folder, file)
            img = preprocess_image(img_path)
            X.append(img)
            y.append(idx)
            
            # Аугментации для каждого оригинала
            for a in augment_image(img, idx):  # Передаем индекс класса
                X.append(a)
                y.append(idx)
    
    return np.array(X), np.array(y), class_map

def save_npy_files(X, y, class_map, features_dir="features"):
    os.makedirs(features_dir, exist_ok=True)
    
    np.save(os.path.join(features_dir, "X.npy"), X)
    np.save(os.path.join(features_dir, "y.npy"), y)
    with open(os.path.join("", "class_map.npy"), 'wb') as f:
        np.save(f, class_map)

def load_npy_files(features_dir="features"):
    X_path = os.path.join(features_dir, "X.npy")
    y_path = os.path.join(features_dir, "y.npy")
    class_map_path = os.path.join(features_dir, "class_map.npy")
    
    if os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(class_map_path):
        X = np.load(X_path)
        y = np.load(y_path)
        with open(class_map_path, 'rb') as f:
            class_map = np.load(f, allow_pickle=True).item()
        return X, y, class_map
    else:
        return None, None, None

def save_features(features, model_name, dataset_split, features_dir="features"):
    os.makedirs(features_dir, exist_ok=True)
    
    np.save(os.path.join(features_dir, f"{model_name}_{dataset_split}_features.npy"), features)

def load_features(model_name, dataset_split, features_dir="features"):
    features_path = os.path.join(features_dir, f"{model_name}_{dataset_split}_features.npy")
    
    if os.path.exists(features_path):
        return np.load(features_path)
    else:
        return None

def extract_features(model, X, preprocess_fn, batch_size=32):
    features = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_prep = preprocess_fn(batch * 255.0)
        feats = model.predict(batch_prep, verbose=0)
        features.extend([f.flatten() for f in feats])
        gc.collect()
    return np.array(features)

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
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()
    plt.show()

def main():
    X, y, class_map = load_npy_files()
    
    if X is None or y is None or class_map is None:
        X, y, class_map = load_dataset(DATASET_PATH)
        save_npy_files(X, y, class_map)

    # Разделение на обучающие, валидационные и тестовые данные
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.2, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=SEED)

    # Удаляем исходные данные из памяти после разделения
    del X, y, X_temp, y_temp
    gc.collect()

    # Загружаем признаки, если они уже сохранены
    logger.info("Загрузка признаков для VGG16 и Xception...")

    vgg_train = load_features("vgg", "train")
    vgg_val = load_features("vgg", "val")
    vgg_test = load_features("vgg", "test")
    xcep_train = load_features("xcep", "train")
    xcep_val = load_features("xcep", "val")
    xcep_test = load_features("xcep", "test")

    if vgg_train is None or vgg_val is None or vgg_test is None or xcep_train is None or xcep_val is None or xcep_test is None:
        logger.info("Извлечение признаков VGG16...")
        vgg = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        vgg_model = Model(inputs=vgg.input, outputs=vgg.output)
        vgg_train = extract_features(vgg_model, X_train, vgg_preprocess)
        vgg_val = extract_features(vgg_model, X_val, vgg_preprocess)
        vgg_test = extract_features(vgg_model, X_test, vgg_preprocess)
        save_features(vgg_train, "vgg", "train")
        save_features(vgg_val, "vgg", "val")
        save_features(vgg_test, "vgg", "test")

        # Удаляем модель VGG16 из памяти
        del vgg, vgg_model
        gc.collect()

        logger.info("Извлечение признаков Xception...")
        xcep = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        xcep_model = Model(inputs=xcep.input, outputs=xcep.output)
        xcep_train = extract_features(xcep_model, X_train, xcep_preprocess)
        xcep_val = extract_features(xcep_model, X_val, xcep_preprocess)
        xcep_test = extract_features(xcep_model, X_test, xcep_preprocess)
        save_features(xcep_train, "xcep", "train")
        save_features(xcep_val, "xcep", "val")
        save_features(xcep_test, "xcep", "test")

        # Удаляем модель Xception из памяти
        del xcep, xcep_model
        gc.collect()

    # Удаляем исходные данные после извлечения признаков
    del X_train, X_val, X_test
    gc.collect()

    # Объединяем признаки
    X_train_comb = np.concatenate([vgg_train, xcep_train], axis=1)
    X_val_comb = np.concatenate([vgg_val, xcep_val], axis=1)
    X_test_comb = np.concatenate([vgg_test, xcep_test], axis=1)

    # Удаляем признаки после объединения
    del vgg_train, vgg_val, vgg_test, xcep_train, xcep_val, xcep_test
    gc.collect()

    logger.info("Обучение XGBoost модели...")
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.2,
        random_state=SEED,
        tree_method='hist',
        device='cuda',
        n_jobs=-1
    )
    clf.fit(X_train_comb, y_train, eval_set=[(X_val_comb, y_val)], verbose=True)

    # Удаляем объединенные признаки после обучения
    del X_train_comb, X_val_comb
    gc.collect()

    logger.info("Оценка модели...")
    y_pred = clf.predict(X_test_comb)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    #Вывод отчета о классификации
    report = classification_report(y_test, y_pred, target_names=class_map.keys())
    print("\nClassification Report:")
    print(report)

    # Удаляем тестовые данные после оценки
    del X_test_comb, y_test, y_pred
    gc.collect()

    joblib.dump(clf, "xgb_model.pkl")
    logger.info("✅ Готово! Модель сохранена как xgb_model.pkl")

    # Удаляем модель XGBoost из памяти
    del clf
    gc.collect()

if __name__ == "__main__":
    main()
