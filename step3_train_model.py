from clearml import Task
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ClearML Task
task = Task.init(
    project_name='Agri-Pest-Detection',
    task_name='Step 3 - Model Training'
)

# Connect parameters
params = task.connect({
    'step2_task_id': 'a76d8cec491f4d75ac38d9befe59df92',  # Will be filled by pipeline or manually
    'processed_dataset_id': 'a76d8cec491f4d75ac38d9befe59df92',  # Alternative way to specify Step 2 task
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'dropout_rate': 0.5,
    'test_queue': 'pipeline',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
})

# Execute remotely if a test_queue is specified (comment out if running locally)
# if params.get('test_queue'):
#     task.execute_remotely(queue_name=params['test_queue'])

# Get Step 2 task ID
step2_task_id = params.get('step2_task_id') or params.get('processed_dataset_id')
if not step2_task_id:
    raise ValueError("Either 'step2_task_id' or 'processed_dataset_id' must be provided")

logger.info(f"Using Step 2 task ID: {step2_task_id}")
logger.info(f"Training configuration:")
for key, value in params.items():
    logger.info(f"  {key}: {value}")

# Load preprocessed data from Step 2
step2_task = Task.get_task(task_id=step2_task_id)
X_train = np.load(step2_task.artifacts['X_train.npy'].get_local_copy())
X_test = np.load(step2_task.artifacts['X_test.npy'].get_local_copy())
y_train = np.load(step2_task.artifacts['y_train.npy'].get_local_copy())
y_test = np.load(step2_task.artifacts['y_test.npy'].get_local_copy())
label2id = np.load(step2_task.artifacts['label2id.npy'].get_local_copy(), allow_pickle=True).item()

logger.info(f"Loaded data - Train: {X_train.shape}, Test: {X_test.shape}")
logger.info(f"Number of classes: {len(label2id)}")

# Normalize data
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

# Define CNN model
class PestCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(PestCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize model
device = torch.device(params['device'])
model = PestCNN(num_classes=len(label2id), dropout_rate=params['dropout_rate']).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return total_loss / len(loader), correct / total

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return total_loss / len(loader), correct / total

# Training loop
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

logger.info(f"Starting training for {params['num_epochs']} epochs...")
best_val_acc = 0
best_epoch = 0

for epoch in range(params['num_epochs']):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Track best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        # Save best model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc,
            'val_loss': val_loss
        }, 'best_model_checkpoint.pth')
    
    logger.info(f'Epoch {epoch+1}/{params["num_epochs"]} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Report metrics to ClearML - IMPORTANT for HPO
    task.get_logger().report_scalar(title="loss", series="train", iteration=epoch, value=train_loss)
    task.get_logger().report_scalar(title="loss", series="validation", iteration=epoch, value=val_loss)
    task.get_logger().report_scalar(title="accuracy", series="train", iteration=epoch, value=train_acc)
    task.get_logger().report_scalar(title="accuracy", series="validation", iteration=epoch, value=val_acc)
    
    # Also report as single values for easier access
    task.get_logger().report_single_value(name="last_train_accuracy", value=train_acc)
    task.get_logger().report_single_value(name="last_val_accuracy", value=val_acc)
    task.get_logger().report_single_value(name="best_val_accuracy", value=best_val_acc)
    task.get_logger().report_single_value(name="best_epoch", value=best_epoch)

# Final evaluation on test set
test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

# Calculate additional metrics
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        _, predicted = output.max(1)
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.numpy())

precision = precision_score(all_targets, all_predictions, average='macro')
recall = recall_score(all_targets, all_predictions, average='macro')

logger.info(f"\nFinal Test Results:")
logger.info(f"Test Loss: {test_loss:.4f}")
logger.info(f"Test Accuracy: {test_accuracy:.4f}")
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall: {recall:.4f}")
logger.info(f"Best Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")

# Report final metrics
task.get_logger().report_single_value(name="test_accuracy", value=test_accuracy)
task.get_logger().report_single_value(name="test_precision", value=precision)
task.get_logger().report_single_value(name="test_recall", value=recall)

# Save the final model
torch.save({
    'epoch': params['num_epochs'],
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'test_accuracy': test_accuracy,
    'test_precision': precision,
    'test_recall': recall,
    'best_val_accuracy': best_val_acc,
    'hyperparameters': params
}, 'pest_cnn_model.pth')
task.upload_artifact('model', artifact_object='pest_cnn_model.pth')
task.upload_artifact('best_model', artifact_object='best_model_checkpoint.pth')

# Plot training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best Val Acc: {best_val_acc:.4f}')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
task.upload_artifact('training_curves', artifact_object='training_curves.png')

logger.info("✅ Model training completed successfully!")
