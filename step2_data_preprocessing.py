from clearml import Task
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image

# Initialize ClearML Task
task = Task.init(
    project_name='Agri-Pest-Detection',
    task_name='Step 2 - Preprocessing (artifact version)'
)
task.execute_remotely()  # Optional: Submit to ClearML Agent

# Connect parameters (Task ID from Step 1)
params = task.connect({
    'dataset_task_id': '',  # ← Replace with actual Task ID from Step 1
    'image_size': 256,
    'test_size': 0.2,
    'random_state': 42
})

# === Step 2.1: Retrieve local path from Step 1 artifact ===
step1_task = Task.get_task(task_id=params['dataset_task_id'])
local_path = step1_task.artifacts['image_dataset_dir'].get_local_copy()

# === Step 2.2: Assuming structure is image_dataset_dir/train/class_name/image.jpg ===
train_dir = os.path.join(local_path, 'train')
if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"❌ 'train' directory not found: {train_dir}")

# === Step 2.3: Image preprocessing ===
img_size = (params['image_size'], params['image_size'])
x_data, y_data = [], []

# Get class labels
dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
label2id = {name: idx for idx, name in enumerate(dirs)}

for class_name in dirs:
    class_dir = os.path.join(train_dir, class_name)
    for img_file in glob(os.path.join(class_dir, '*.jpg')):
        try:
            img = Image.open(img_file).convert('RGB').resize(img_size)
            x_data.append(np.array(img))
            y_data.append(label2id[class_name])
        except Exception as e:
            print(f"⚠️ Skipping image {img_file}, error: {e}")
            continue

x_data = np.array(x_data)
y_data = np.array(y_data)

# === Step 2.4: Split into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=params['test_size'], random_state=params['random_state']
)

# === Step 2.5: Save as local .npy files ===
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('label2id.npy', label2id)

# === Step 2.6: Upload as ClearML artifacts (for Step 3 use) ===
task.upload_artifact("X_train.npy", artifact_object='X_train.npy')
task.upload_artifact("X_test.npy", artifact_object='X_test.npy')
task.upload_artifact("y_train.npy", artifact_object='y_train.npy')
task.upload_artifact("y_test.npy", artifact_object='y_test.npy')
task.upload_artifact("label2id.npy", artifact_object='label2id.npy')

print("✅ Preprocessing complete. 5 .npy files uploaded as artifacts, ready for Step 3.")
