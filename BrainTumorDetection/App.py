# INSTALLATION
# !pip install torch torchvision numpy pandas matplotlib opencv-python scikit-learn seaborn tensorflow keras

# IMPORTS
from pathlib import Path
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import ToPILImage

import tensorflow as tf
from tensorflow.keras import models as kmodels
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import DenseNet121, EfficientNetB0
from tensorflow.keras.optimizers import Adam

# PATH CHECKING
def check_path_exists(path):
    return Path(path).exists()

path_to_check = 'Brain_tumor_dataset/Testing/glioma/Te-gl_0011.jpg'
print(check_path_exists(path_to_check))

# DATA READING
dataset_path = Path('Brain_tumor_dataset/')
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
data = []

for phase in ['Training', 'Testing']:
    phase_path = dataset_path / phase
    if phase_path.is_dir():
        for class_name in classes:
            class_path = phase_path / class_name
            if class_path.is_dir():
                for img_path in class_path.iterdir():
                    if img_path.suffix in ['.jpg', '.png']:
                        data.append((img_path, class_name, phase))
            else:
                print(f"Class directory does not exist: {class_path}")
    else:
        print(f"Phase directory does not exist: {phase_path}")

df = pd.DataFrame(data, columns=['image_path', 'label', 'phase'])
train_df = df[df['phase'] == 'Training']
test_df = df[df['phase'] == 'Testing']

print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

# TORCH DATASET AND DATALOADER
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class BrainTumorDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = classes.index(self.dataframe.iloc[idx, 1])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = BrainTumorDataset(train_df, transform=transform)
test_dataset = BrainTumorDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")

# DATA VISUALIZATION
def plot_class_distribution(df, title):
    class_counts = df['label'].value_counts()
    labels = class_counts.index
    sizes = class_counts.values
    explode = [0.1] * len(labels)
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')
    plt.show()

plot_class_distribution(train_df, 'Training Data Class Distribution')
plot_class_distribution(test_df, 'Testing Data Class Distribution')


# TENSORFLOW DATA GENERATOR
image_size = (150, 150)
batch_size = 32
SEED = 42

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    brightness_range=(0.85, 1.15),
    width_shift_range=0.002,
    height_shift_range=0.002,
    shear_range=12.5,
    zoom_range=0,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="nearest"
)

train_generator = train_datagen.flow_from_directory(
    'Brain_tumor_dataset/Training/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    seed=SEED
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'Brain_tumor_dataset/Testing/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

class_indices_train = train_generator.class_indices
print("Categorical types for the training data:")
print(class_indices_train)

# MODEL DEFINITIONS
def create_basic_cnn_model(image_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=image_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def create_advanced_cnn_model(image_shape, num_classes):
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=image_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def create_densenet_model(image_shape, num_classes):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=image_shape)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model


def create_dgnet_model(image_shape, num_classes):
    model = Sequential([
        Conv2D(64, (7, 7), activation='relu', input_shape=image_shape),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def create_efficientnet_model(image_shape, num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_shape[0], image_shape[1], 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model

# TRAINING FUNCTION
def compile_and_train_model(model, train_generator, test_generator, epochs=30):
    model.compile(
        optimizer=Adam(learning_rate=0.002),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(train_generator, epochs=epochs, validation_data=test_generator)
    return history

# TRAIN MODELS
image_shape = (150, 150, 3)
num_classes = len(train_generator.class_indices)

print("Training Basic CNN Model...")
basic_cnn_model = create_basic_cnn_model(image_shape, num_classes)
basic_cnn_history = compile_and_train_model(basic_cnn_model, train_generator, test_generator)

print("Training Advanced CNN Model...")
advanced_cnn_model = create_advanced_cnn_model(image_shape, num_classes)
advanced_cnn_history = compile_and_train_model(advanced_cnn_model, train_generator, test_generator, epochs=30)

print("Training DenseNet Model...")
densenet_model = create_densenet_model(image_shape, num_classes)
densenet_history = compile_and_train_model(densenet_model, train_generator, test_generator, epochs=30)

print("Training DGNet Model...")
dgnet_model = create_dgnet_model(image_shape, num_classes)
dgnet_history = compile_and_train_model(dgnet_model, train_generator, test_generator, epochs=30)

# EVALUATION FUNCTION
def evaluate_model(model, test_generator, class_labels):
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    return accuracy, cm, predicted_classes

# EVALUATE ALL MODELS
class_labels = list(class_indices_train.keys())
print("Evaluating Basic CNN Model")
evaluate_model(basic_cnn_model, test_generator, class_labels)

print("Evaluating Advanced CNN Model")
evaluate_model(advanced_cnn_model, test_generator, class_labels)

print("Evaluating DenseNet Model")
evaluate_model(densenet_model, test_generator, class_labels)

print("Evaluating DGNet Model")
evaluate_model(dgnet_model, test_generator, class_labels)

# PREDICTION ON SINGLE IMAGE
def predict_image(model, img_path, class_labels):
    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = class_labels[predicted_class[0]]
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}")
    plt.show()
    return predicted_label

# Example Prediction
img_path = 'Brain_tumor_dataset/Testing/glioma/Te-gl_0011.jpg'
predicted_label = predict_image(advanced_cnn_model, img_path, class_labels)
print(f"Predicted Label: The image shows that it comes under {predicted_label}")

# SAVE MODELS
basic_cnn_model.save('basic_cnn_model.h5')
advanced_cnn_model.save('advanced_cnn_model.h5')
densenet_model.save('densenet_model.h5')
dgnet_model.save('dgnet_model.h5')

print("All models saved successfully.")
