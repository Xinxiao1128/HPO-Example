import os
from clearml import Task, Dataset

# Initialize the ClearML Task
task = Task.init(
    project_name="examples",
    task_name="Step 1 - Load Uncompressed Pest Image Dataset"
)

task.execute_remotely()  # Optional: submit to ClearML Agent

# Load Dataset using name + version (recommended)
dataset_name = "Pest Image Dataset"
dataset_version = "v1.0"

# Load Dataset (more stable than using ID)
print(f"🔍 Loading Dataset: {dataset_name} v{dataset_version}")
dataset = Dataset.get(dataset_name=dataset_name, dataset_version=dataset_version)
dataset_path = dataset.get_local_copy()
print(f"✅ Dataset retrieved, local path: {dataset_path}")

# Check if it contains a 'train' subdirectory (ImageFolder structure)
expected_subdir = os.path.join(dataset_path, "train")
if not os.path.isdir(expected_subdir):
    print("⚠️ Warning: 'train/' subdirectory not found. Please verify the dataset structure or adjust the paths accordingly.")
else:
    print(f"📁 'train' directory detected: {expected_subdir}")

# Upload dataset path as an artifact (for Step 2 use)
task.upload_artifact(name="image_dataset_dir", artifact_object=dataset_path)
print("✅ Successfully uploaded artifact: image_dataset_dir ✅")
