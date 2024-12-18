import tensorflow as tf
from tensorflow.keras import layers, models                                             # type: ignore
import matplotlib.pyplot as plt
train_dir = 'F:\\3rd SEM PG\\Deep_Learning_LAB\\04\\Binary_Classifier\\Try03(Final)\\img\\train'                         # Define paths to training and test data
test_dir = 'F:\\3rd SEM PG\\Deep_Learning_LAB\\04\\Binary_Classifier\\Try03(Final)\\img\\test'
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(                    # Load and preprocess images
    train_dir, image_size=(128, 128), batch_size=64, shuffle=True)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=(128, 128), batch_size=64)
model = models.Sequential([                                                             # Define CNN model
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(),layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(), layers.Flatten(),layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')])                                             # Binary classification
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])       # Compile and train the model
history = model.fit(train_dataset, epochs=30, validation_data=test_dataset)
plt.figure(figsize=(6, 8))                                                              # Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs'),plt.ylabel('Accuracy'),plt.show()
test_loss, test_acc = model.evaluate(test_dataset)                                      # Evaluate the model
print(f'Test Accuracy: {test_acc * 100:.2f}%')
def load_and_preprocess_sample_image(filepath):                                         # Load and preprocess a sample image for prediction
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (128, 128)) / 255.0                                  # Normalize
    return tf.expand_dims(image, axis=0)                                                # Add batch dimension
sample_image_path = 'F:\\3rd SEM PG\\Deep_Learning_LAB\\04\\Test.jpeg'   # Sample image path and prediction
sample_image = load_and_preprocess_sample_image(sample_image_path)
predictions = model.predict(sample_image)
predicted_class = 'Car' if predictions[0] > 0.5 else 'Bike'
print(f'The given image is: {predicted_class}')
plt.imshow(tf.squeeze(sample_image))                                                    # Use sample_image instead of sample_image_path
plt.axis('off'),plt.show()  # Displays the image