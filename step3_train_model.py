from clearml import Task, Dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Disable GPU
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==== (1) Initialize ClearML Task ====
task = Task.init(project_name='Agri-Pest-Detection', task_name='Step 3 - Model Training')
task.execute_remotely()  # Optional: Run remotely on ClearML Agent

# ==== (2) Parameter Setup ====
params = task.connect({
    'step2_task_id': '',  # ✅ Replace with actual Step 2 Task ID
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.0001
})

# ==== (3) Load output dataset from Step 2 ====
step2_task = Task.get_task(task_id=params['step2_task_id'])
X_train = np.load(step2_task.artifacts['X_train.npy'].get_local_copy())
X_test = np.load(step2_task.artifacts['X_test.npy'].get_local_copy())
y_train = np.load(step2_task.artifacts['y_train.npy'].get_local_copy())
y_test = np.load(step2_task.artifacts['y_test.npy'].get_local_copy())
label2id = np.load(step2_task.artifacts['label2id.npy'].get_local_copy(), allow_pickle=True).item()

# ==== (4) Build CNN model ====
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label2id), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==== (5) Train the model ====
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=params['epochs'],
    batch_size=params['batch_size']
)

# ==== (6) Save model and training plots ====
model.save('cnn_model.h5')
task.upload_artifact("model", artifact_object='cnn_model.h5')

# Accuracy curve
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("accuracy_curve.png")
task.upload_artifact("accuracy_curve", artifact_object='accuracy_curve.png')

# Loss curve
plt.clf()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_curve.png")
task.upload_artifact("loss_curve", artifact_object='loss_curve.png')

# ==== (7) Print evaluation metrics ====
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')

print(f"Accuracy  : {acc * 100:.2f}%")
print(f"Precision : {prec * 100:.2f}%")
print(f"Recall    : {rec * 100:.2f}%")

print("✅ Model training completed. Metrics printed, model and plots uploaded to ClearML.")
