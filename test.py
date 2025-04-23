import os
import numpy as np
import joblib
from PIL import Image
import cv2
import random
from tensorflow.keras.applications import VGG16, Xception
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

IMG_SIZE = 200
#loading_str = f'_size_{IMG_SIZE}'
loading_str = ''
TEST_FOLDER = 'grape_leaf_dataset'
SAMPLE_RATIO = 0.1  # 10%
SEED = 43  # любое фиксированное число
random.seed(SEED)
np.random.seed(SEED)


# Загрузка модели и class map
model = joblib.load(f"xgb_model{loading_str}.pkl")
class_map = np.load("class_map.npy", allow_pickle=True).item()
id_to_class = {v: k for k, v in class_map.items()}
class_to_id = {v: k for k, v in id_to_class.items()}

# Предобученные модели
vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
vgg_model = Model(inputs=vgg_model.input, outputs=vgg_model.output)

xception_model = Xception(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
xception_model = Model(inputs=xception_model.input, outputs=xception_model.output)

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

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        img = remove_background(image)
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return resized / 255.0
    except Exception as e:
        print(f"Ошибка обработки {image_path}: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE, 3))

def test_sample():
    total = 0
    correct = 0
    all_preds = []
    all_true = []

    for class_folder in os.listdir(TEST_FOLDER):
        class_path = os.path.join(TEST_FOLDER, class_folder)
        if not os.path.isdir(class_path):
            continue

        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            continue

        sample_size = max(1, int(len(image_files) * SAMPLE_RATIO))
        sample_files = random.sample(image_files, sample_size)

        for image_name in sample_files:
            image_path = os.path.join(class_path, image_name)
            image = preprocess_image(image_path)
            image = np.expand_dims(image, axis=0)

            vgg_features = vgg_model.predict(image)
            xception_features = xception_model.predict(image)
            combined_features = np.concatenate([vgg_features.flatten(), xception_features.flatten()]).reshape(1, -1)

            try:
                prediction = model.predict(combined_features)[0]
                predicted_class = id_to_class[prediction]
                is_correct = (predicted_class == class_folder)
                correct += int(is_correct)
                total += 1
                all_preds.append(prediction)
                all_true.append(class_to_id[class_folder])
                print(f"[{class_folder}] {image_name} → Предсказано: {predicted_class} {'✅' if is_correct else '❌'}")
            except Exception as e:
                print(f"[{class_folder}] {image_name} → Ошибка: {e}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nТочность (accuracy): {accuracy:.2%} ({correct}/{total})")

    if all_preds and all_true:
        print("\n📊 Classification Report:")
        print(classification_report(all_true, all_preds, target_names=list(class_map.keys())))

        cm = confusion_matrix(all_true, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_map.keys()))
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title("Матрица ошибок (Confusion Matrix)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_sample()
