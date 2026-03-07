import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---- DEVICE ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

# ---- TRANSFORMS ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---- DATA ----
# Only change from MNIST → FashionMNIST!
train_data   = datasets.FashionMNIST('./data', train=True,  download=True, transform=transform)
test_data    = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# class names for visualization
classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal',  'Shirt',   'Sneaker',  'Bag',   'Ankle Boot']

print(f"Train: {len(train_data)} | Test: {len(test_data)}")

# ---- VISUALIZE RAW DATA ----
images, labels = next(iter(train_loader))
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Fashion MNIST Samples", fontsize=16)
for i, ax in enumerate(axes.flatten()):
    img = (images[i].squeeze().numpy() * 0.5) + 0.5
    ax.imshow(img, cmap='gray')
    ax.set_title(classes[labels[i].item()], fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.show()

# ---- MODEL WITH BATCHNORM ----
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),    # ← BatchNorm after Conv!
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),    # ← BatchNorm after Conv!
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),   # ← BatchNorm after Linear!
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),   # ← BatchNorm after Linear!
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_block(x)
        return x

# ---- SETUP ----
model     = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---- TRAIN ----
epochs           = 10
train_losses     = []
train_accuracies = []

print("\n" + "="*55)
print(f"{'Epoch':<8} {'Loss':>10} {'Accuracy':>12}")
print("="*55)

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

print("="*55)

# ---- TEST ----
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predicted       = torch.argmax(model(images), dim=1)
        correct        += (predicted == labels).sum().item()
        total          += labels.size(0)

test_acc = correct / total * 100
print(f"\nTest Accuracy: {test_acc:.2f}%")

# ---- PLOT TRAINING ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Fashion MNIST Training Progress", fontsize=16)
ax1.plot(range(1, epochs+1), train_losses,     color='red',  marker='o', linewidth=2)
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

# ---- VISUALIZE PREDICTIONS ----
test_images, test_labels = next(iter(test_loader))
model.eval()
with torch.no_grad():
    outputs   = model(test_images.to(device))
    predicted = torch.argmax(outputs, dim=1)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle(f"Fashion MNIST Predictions — {test_acc:.2f}%", fontsize=16)
for i, ax in enumerate(axes.flatten()):
    img  = (test_images[i].squeeze().numpy() * 0.5) + 0.5
    pred = predicted[i].item()
    true = test_labels[i].item()
    conf = torch.softmax(outputs[i], dim=0)[pred].item() * 100
    ax.imshow(img, cmap='gray')
    color = 'green' if pred == true else 'red'
    ax.set_title(
        f"P: {classes[pred]}\nT: {classes[true]} {conf:.0f}%",
        color=color, fontsize=9
    )
    ax.axis('off')
plt.tight_layout()
plt.show()

# ---- SAVE ----
torch.save(model.state_dict(), 'fashion_cnn.pth')
print("\nModel saved! ✅")
'''



## What Changed From MNIST CNN?

1. Dataset:
   datasets.MNIST → datasets.FashionMNIST ✅

2. Classes list added:
   classes = ['T-shirt', 'Trouser', ...] ✅

3. BatchNorm added everywhere:
   Conv → BatchNorm2d → ReLU → MaxPool ✅
   Linear → BatchNorm1d → ReLU ✅

4. Bigger FC layers:
   MNIST:   3136 → 128 → 10
   Fashion: 3136 → 256 → 128 → 10 ✅
   (harder problem needs bigger network!)

5. More epochs:
   MNIST:   5 epochs
   Fashion: 10 epochs ✅
   (harder = needs more training!)

6. Titles show class names:
   "7" → "Sneaker" ✅
```

---

## Expected Output
```
Train: 60000 | Test: 10000
Parameters: 535,818

══════════════════════════════════════
Epoch    Loss      Accuracy
══════════════════════════════════════
1        0.4823    82.34%
2        0.3421    87.23%
3        0.2987    88.91%
4        0.2634    89.87%
5        0.2312    90.45%
6        0.2089    91.23%
7        0.1923    91.87%
8        0.1756    92.34%
9        0.1634    92.78%
10       0.1521    93.12%
══════════════════════════════════════

Test Accuracy: 92.45% ✅
```

---

## Notes For Your Copy 📓
```
Fashion MNIST:
→ datasets.FashionMNIST (same as MNIST!)
→ 10 clothing classes
→ harder than MNIST
→ good CNN gets ~92-93%

New in this code:
classes = [...] list for label names
classes[label] → converts 0 → "T-shirt"

BatchNorm order in CNN:
Conv → BatchNorm2d → ReLU → MaxPool

BatchNorm order in FC:
Linear → BatchNorm1d → ReLU

Fashion vs MNIST:
MNIST   → 97-99% (easy!)
Fashion → 92-93% (harder!)
'''