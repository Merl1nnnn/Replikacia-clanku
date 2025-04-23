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
import random

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
IMG_SIZE = 128
DATASET_PATH = "grape_leaf_dataset"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
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

        # 🔹 Приёмирование: просто сглаживаем яркость через размытие
        img = cv2.blur(img, (3, 3))  # усреднение по 3x3 окну

        # 🔸 Билатеральная фильтрация: мягкое сглаживание с сохранением границ
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

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

def load_dataset(path, max_per_class=500, max_per_difficult_class=1000, test_ratio=0.15, 
                 train_file='train.txt', val_file='val.txt', test_file='test.txt',
                 augment_count=1):  # Сколько аугментированных копий делать
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    class_map = {name: idx for idx, name in enumerate(sorted(os.listdir(path)))}
    logger.info("Классы: %s", class_map)

    train_paths, val_paths, test_paths = [], [], []

    for label, idx in class_map.items():
        folder = os.path.join(path, label)
        images = os.listdir(folder)
        np.random.shuffle(images)

        # Ограничения по числу изображений
        if idx in [0, 1]:
            images = images[:max_per_difficult_class]
        elif idx in [2, 3]:
            images = images[:max_per_class]
        else:
            logger.warning(f"Класс {idx} не обработан, пропускаем...")
            continue

        # Разделяем на train / val / test
        num_test = int(len(images) * test_ratio)
        num_val = int(len(images) * 0.15)
        test_images = images[:num_test]
        val_images = images[num_test:num_test + num_val]
        train_images = images[num_test + num_val:]

        # Обработка и аугментация
        def process_and_append(img_list, X, y, paths, label_idx, augment=False):
            for img_name in img_list:
                img_path = os.path.join(folder, img_name)
                base_img = preprocess_image(img_path)
                X.append(base_img)
                y.append(label_idx)
                paths.append(img_path)

                # Аугментация только для train
                if augment:
                    aug_images = augment_image(base_img, label_idx)
                    for aug in aug_images[:augment_count]:
                        X.append(aug)
                        y.append(label_idx)

        process_and_append(train_images, X_train, y_train, train_paths, idx, augment=True)
        process_and_append(val_images, X_val, y_val, val_paths, idx, augment=False)
        process_and_append(test_images, X_test, y_test, test_paths, idx, augment=False)

    # Сохраняем пути
    def save_paths(paths, filename):
        with open(filename, 'w') as f:
            for path in paths:
                f.write(path + '\n')

    save_paths(train_paths, train_file)
    save_paths(val_paths, val_file)
    save_paths(test_paths, test_file)

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_val), np.array(y_val),
        np.array(X_test), np.array(y_test),
        class_map
    )

def extract_features(model, X, preprocess_fn, batch_size=32):
    features = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_prep = preprocess_fn(batch * 255.0)

        # Убедитесь, что используете GPU
        with tf.device('/GPU:0'):
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

def main():
    # Загружаем данные
    X_train, y_train, X_val, y_val, X_test, y_test, class_map = load_dataset(DATASET_PATH)

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"✅ GPU доступен: {gpus}")
    else:
        print("⚠️ GPU не найден. Используется CPU.")
    tf.config.set_visible_devices(gpus[0], 'GPU')

    # Сохраняем class_map в текстовый файл
    logger.info("Сохраняем class_map в файл...")
    np.save('class_map.npy', class_map)

    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    xcep = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    with tf.device('/GPU:0'):  # /GPU:0 — первый доступный GPU
        vgg_model = Model(inputs=vgg.input, outputs=vgg.output)
        xcep_model = Model(inputs=xcep.input, outputs=xcep.output)

    logger.info("Извлечение признаков VGG16...")
    vgg_train = extract_features(vgg_model, X_train, vgg_preprocess)
    vgg_val = extract_features(vgg_model, X_val, vgg_preprocess)
    vgg_test = extract_features(vgg_model, X_test, vgg_preprocess)

    # Удаляем модель VGG16 из памяти
    del vgg, vgg_model
    gc.collect()

    logger.info("Извлечение признаков Xception...")
    xcep_train = extract_features(xcep_model, X_train, xcep_preprocess)
    xcep_val = extract_features(xcep_model, X_val, xcep_preprocess)
    xcep_test = extract_features(xcep_model, X_test, xcep_preprocess)

    # Удаляем модель Xception из памяти
    del xcep, xcep_model
    gc.collect()

    # Объединяем признаки
    X_train_comb = np.concatenate([vgg_train, xcep_train], axis=1)
    X_val_comb = np.concatenate([vgg_val, xcep_val], axis=1)
    X_test_comb = np.concatenate([vgg_test, xcep_test], axis=1)

    # Удаляем признаки после объединения
    del vgg_train, vgg_val, vgg_test, xcep_train, xcep_val, xcep_test
    gc.collect()

    # Обучение XGBoost модели
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

    # Оценка модели
    logger.info("Оценка модели...")
    y_pred = clf.predict(X_test_comb)
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:")
    logger.info(cm)

    # Вывод отчета о классификации
    report = classification_report(y_test, y_pred, target_names=class_map.keys())
    logger.info("\nClassification Report:")
    logger.info(report)

    # Сохранение модели
    joblib.dump(clf, "xgb_model.pkl")
    logger.info("✅ Готово! Модель сохранена как xgb_model.pkl")
    
    # Удаляем модель XGBoost из памяти
    del clf
    
    gc.collect()

if __name__ == "__main__":
    main()
