import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# MNIST データセットを読み込む
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# データの前処理
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# ラベルを one-hot エンコーディングする
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# ニューラルネットワークモデルを作る
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# モデルをコンパイルする
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# モデルを学習する
history = model.fit(train_images, train_labels, epochs=20, batch_size=128, validation_split=0.1)

# モデルを評価する
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'\nTest accuracy: {test_acc:.4f}')

# 損失と精度をプロットする
plt.figure(figsize=(12, 4))

# 損失をプロット
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 精度をプロット
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('mnist_training.png', dpi=300)

plt.show()


# モデルを使って画像を識別する
def predict_and_plot_image(model, test_images, test_labels):
    # テスト画像からランダムに1枚を選ぶ
    index = np.random.randint(0, len(test_images))
    test_image = test_images[index]
    test_label = test_labels[index]

    # 画像を予測する
    predictions = model.predict(np.expand_dims(test_image, axis=0))
    predicted_label = np.argmax(predictions[0])

    # 実際のラベルを取得する
    actual_label = np.argmax(test_label)

    # 画像を表示する
    plt.figure(figsize=(6, 6))
    plt.imshow(test_image.squeeze(), cmap='gray')
    plt.title(f'Predicted: {predicted_label}, Actual: {actual_label}')
    plt.axis('off')
    plt.savefig('predict.png', dpi=300)
    plt.show()


# 画像の予測と表示を行う
predict_and_plot_image(model, test_images, test_labels)
