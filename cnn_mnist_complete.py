import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# TRANSFORMS
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# DATA
train_data   = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data    = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# MODEL
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_block(x)
        return x

# SETUP
model     = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# TRAIN
epochs           = 5
train_losses     = []
train_accuracies = []

print("\n" + "="*50)
print(f"{'Epoch':<8} {'Loss':>10} {'Accuracy':>12}")
print("="*50)

for epoch in range(epochs):
    model.train()
    total_loss = correct = total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs        = model(images)
        loss           = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted   = torch.argmax(outputs, dim=1)
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total * 100
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f"{epoch+1:<8} {avg_loss:>10.4f} {accuracy:>11.2f}%")

print("="*50)

# TEST
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predicted       = torch.argmax(model(images), dim=1)
        correct        += (predicted == labels).sum().item()
        total          += labels.size(0)

test_acc = correct / total * 100
print(f"\nTest Accuracy : {test_acc:.2f}%")
print(f"ANN was       : 97.65%")
print(f"CNN is        : {test_acc:.2f}%")

# PLOT TRAINING
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("CNN Training Progress", fontsize=16)
ax1.plot(range(1, epochs+1), train_losses, color='red', marker='o', linewidth=2)
ax1.set_title("Loss over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)
ax2.plot(range(1, epochs+1), train_accuracies, color='blue', marker='o', linewidth=2)
ax2.set_title("Accuracy over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True)
plt.tight_layout()
plt.show()

# VISUALIZE PREDICTIONS
test_images, test_labels = next(iter(test_loader))
model.eval()
with torch.no_grad():
    outputs   = model(test_images.to(device))
    predicted = torch.argmax(outputs, dim=1)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle(f"CNN Predictions — Accuracy: {test_acc:.2f}%", fontsize=16)
for i, ax in enumerate(axes.flatten()):
    img  = (test_images[i].squeeze().numpy() * 0.5) + 0.5
    pred = predicted[i].item()
    true = test_labels[i].item()
    conf = torch.softmax(outputs[i], dim=0)[pred.item()].item() * 100
    ax.imshow(img, cmap='gray')
    color = 'green' if pred == true else 'red'
    ax.set_title(f"Pred: {pred} | True: {true}\nConf: {conf:.1f}%",
                 color=color, fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.show()

# SAVE
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("\nModel saved as mnist_cnn.pth ✅")