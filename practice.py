import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#Device _____________________________________
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using : {device}")
#transforms _________________________________________
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

])
#data___________________________________________________________________
train_data   = datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
test_data = datasets.CIFAR10('./data', train=False,  download=True, transform=transform)
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

classes = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer',
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
print(f"Train: {len(train_data)} | Test: {len(test_data)}")

# ---- VISUALIZE ----
images,labels = next(iter(train_loader))
fig,axes = plt.subplots(2,5,figsize=(12,5))
for i , ax in enumerate(axes.flatten()):
    img = (images[i].permute(1, 2, 0).numpy() * 0.5) + 0.5
    img = img.clip(0, 1)
    ax.imshow(img)
    ax.set_title(classes[labels[i].item()], fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.show()
#model _______________________________________________________

class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size= 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)


        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size= 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)



        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size= 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self , x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.fc_block(x)
        return x
model = CIFAR_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
#training loop ______________________________________________________
epochs = 18
train_losses = []
train_accuracies = []
print("\n" + "="*55)
print(f"{'Epoch':<8} {'Loss':>10} {'Accuracy':>12}")
print("="*55)
for epoch in range(epochs):
    model.train()
    total_loss = correct = total = 0
    for images,labels in train_loader:
        images,labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Move accuracy calculations inside the batch loop
        predicted = torch.argmax(outputs,dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Calculate average loss and accuracy for the epoch after processing all batches
    if len(train_loader) > 0:
        avg_loss = total_loss / len(train_loader)
    else:
        avg_loss = 0.0 # Default if no batches were processed

    if total > 0:
        avg_acc = (correct / total) * 100
    else:
        avg_acc = 0.0 # Default if no samples were processed

    train_losses.append(avg_loss)      
    train_accuracies.append(avg_acc)   
    print(f"{epoch+1:<8} {avg_loss:>10.4f} {avg_acc:>11.2f}%")

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

#-----plot-------
ffig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("CIFAR-10 Training Progress", fontsize=16)
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

#__visulaise__
test_images,test_labels = next(iter(test_loader))
model.eval()
with torch.no_grad():
    outputs = model(test_images.to(device))
    predicted = torch.argmax(outputs,dim=1)
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle(f"CIFAR-10 Predictions — Test Acc: {test_acc:.2f}%", fontsize=16)
for i, ax in enumerate(axes.flatten()):
    img  = (test_images[i].permute(1, 2, 0).numpy() * 0.5) + 0.5
    img  = img.clip(0, 1)
    pred = predicted[i].item()
    true = test_labels[i].item()
    conf = torch.softmax(outputs[i], dim=0)[pred].item() * 100
    ax.imshow(img)
    color = 'green' if pred == true else 'red'
    ax.set_title(
        f"P: {classes[int(pred)]}\nT: {classes[int(true)]} {conf:.0f}%",
        color=color, fontsize=9
    )
    ax.axis('off')
plt.tight_layout()
plt.show()

# ---- SAVE ----
torch.save(model.state_dict(), 'cifar10_cnn.pth')
print("\nModel saved as cifar10_cnn.pth ✅")