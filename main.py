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

# === LOGGING SETUP ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === GPU SETUP ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info("✅ GPU available: %s", gpus)
else:
    logger.warning("⚠️ GPU not found. Using CPU.")

# === PARAMETERS ===
IMG_SIZE = 128
DATASET_PATH = "grape_leaf_dataset"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# === FUNCTIONS ===

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

        # 🔹 Blurring to smooth brightness
        img = cv2.blur(img, (3, 3))

        # 🔸 Bilateral filtering: smooth with edge preservation
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return resized / 255.0
    except Exception as e:
        logger.error("Error in %s: %s", path, e)
        return np.zeros((IMG_SIZE, IMG_SIZE, 3))

def augment_image(image, class_idx):
    if class_idx in [0, 1]:  # Light augmentation for difficult classes
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True
        )
    else:  # Standard augmentation for others
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
                 augment_count=1):
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    class_map = {name: idx for idx, name in enumerate(sorted(os.listdir(path)))}
    logger.info("Classes: %s", class_map)

    train_paths, val_paths, test_paths = [], [], []

    for label, idx in class_map.items():
        folder = os.path.join(path, label)
        images = os.listdir(folder)
        np.random.shuffle(images)

        # Limit number of images
        if idx in [0, 1]:
            images = images[:max_per_difficult_class]
        elif idx in [2, 3]:
            images = images[:max_per_class]
        else:
            logger.warning(f"Class {idx} not handled, skipping...")
            continue

        # Split into train / val / test
        num_test = int(len(images) * test_ratio)
        num_val = int(len(images) * 0.15)
        test_images = images[:num_test]
        val_images = images[num_test:num_test + num_val]
        train_images = images[num_test + num_val:]

        def process_and_append(img_list, X, y, paths, label_idx, augment=False):
            for img_name in img_list:
                img_path = os.path.join(folder, img_name)
                base_img = preprocess_image(img_path)
                X.append(base_img)
                y.append(label_idx)
                paths.append(img_path)

                if augment:
                    aug_images = augment_image(base_img, label_idx)
                    for aug in aug_images[:augment_count]:
                        X.append(aug)
                        y.append(label_idx)

        process_and_append(train_images, X_train, y_train, train_paths, idx, augment=True)
        process_and_append(val_images, X_val, y_val, val_paths, idx, augment=False)
        process_and_append(test_images, X_test, y_test, test_paths, idx, augment=False)

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
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test, class_map = load_dataset(DATASET_PATH)

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"✅ GPU available: {gpus}")
    else:
        print("⚠️ GPU not found. Using CPU.")
    tf.config.set_visible_devices(gpus[0], 'GPU')

    # Save class_map to file
    logger.info("Saving class_map to file...")
    np.save('class_map.npy', class_map)

    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    xcep = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    with tf.device('/GPU:0'):
        vgg_model = Model(inputs=vgg.input, outputs=vgg.output)
        xcep_model = Model(inputs=xcep.input, outputs=xcep.output)

    logger.info("Extracting VGG16 features...")
    vgg_train = extract_features(vgg_model, X_train, vgg_preprocess)
    vgg_val = extract_features(vgg_model, X_val, vgg_preprocess)
    vgg_test = extract_features(vgg_model, X_test, vgg_preprocess)

    del vgg, vgg_model
    gc.collect()

    logger.info("Extracting Xception features...")
    xcep_train = extract_features(xcep_model, X_train, xcep_preprocess)
    xcep_val = extract_features(xcep_model, X_val, xcep_preprocess)
    xcep_test = extract_features(xcep_model, X_test, xcep_preprocess)

    del xcep, xcep_model
    gc.collect()

    X_train_comb = np.concatenate([vgg_train, xcep_train], axis=1)
    X_val_comb = np.concatenate([vgg_val, xcep_val], axis=1)
    X_test_comb = np.concatenate([vgg_test, xcep_test], axis=1)

    del vgg_train, vgg_val, vgg_test, xcep_train, xcep_val, xcep_test
    gc.collect()

    logger.info("Training XGBoost model...")
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

    logger.info("Evaluating model...")
    y_pred = clf.predict(X_test_comb)
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:")
    logger.info(cm)

    report = classification_report(y_test, y_pred, target_names=class_map.keys())
    logger.info("\nClassification Report:")
    logger.info(report)

    joblib.dump(clf, "xgb_model.pkl")
    logger.info("✅ Done! Model saved as xgb_model.pkl")

    del clf
    gc.collect()

if __name__ == "__main__":
    main()
