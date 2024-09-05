import json
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, \
    Activation, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2

# Setting up the path for image folder and JSON folder
img_folder = r'C:\syuu\pythonProject\tosho_all_linejson\img\tosho_1870_bunkei'
json_folder = r'C:\syuu\pythonProject\tosho_all_linejson\json\tosho_1870_bunkei'

# Initializing empty lists where images and labels will be added
images = []
labels = []

# Checking the list of files in both folders
img_files = os.listdir(img_folder)
json_files = os.listdir(json_folder)

# Loading each image and corresponding JSON file
for img_file, json_file in zip(img_files, json_files):
    img_path = os.path.join(img_folder, img_file)
    json_path = os.path.join(json_folder, json_file)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        boxes = data['boxes']

    img = cv2.imread(img_path)

    # Loop on each box in JSON file
    for box in boxes:
        coordinates = box['quad']
        roi = img[coordinates['y1']:coordinates['y3'], coordinates['x1']:coordinates['x2']]

        if roi.size != 0:
            roi = cv2.resize(roi, (224, 224))
            roi = roi.astype(np.float32) / 255.0

            texts = box['text']

            images.append(roi)
            labels.append(texts)

# Encoding the labels into integers
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Converting the lists into numpy arrays
np_images = np.array(images)
np_labels = np.array(labels_encoded)

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(np_images, np_labels, test_size=0.2)

# Load pre-trained ResNet50 model, discard the top classifier layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classifier layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(np.max(np_labels) + 1, activation='softmax')(x)

# Create entire model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the first few layers from the ResNet50 model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Set callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('ocr_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# Train the model
history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=16,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping, model_checkpoint])

# Load the best model and evaluate performance on the test set
best_model = load_model('ocr_model.keras')
test_loss, test_acc = best_model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
