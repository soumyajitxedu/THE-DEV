# PyTorch — Complete Lock-In Guide (3-Hour Session)

---

## WHAT IS PYTORCH?

**Definition:** PyTorch is an open-source deep learning framework developed by Meta AI (formerly Facebook AI Research). It provides two core things — an N-dimensional tensor library (like NumPy but with GPU support) and automatic differentiation for building and training neural networks.

**Why PyTorch over others?**
PyTorch uses a **dynamic computation graph** (called Define-by-Run), meaning the graph is built on the fly as operations execute. TensorFlow originally used static graphs (Define-then-Run). Dynamic graphs make debugging intuitive — you use normal Python control flow, print statements, and debuggers.

---

## HOUR 1 — TENSORS (The DNA of PyTorch)

---

### 1.1 WHAT IS A TENSOR?

**Definition:** A Tensor is a generalized mathematical object that can represent scalars, vectors, matrices, and higher-dimensional arrays under one unified structure.

```
Scalar  →  0D Tensor  →  just a number, e.g., 5
Vector  →  1D Tensor  →  [1, 2, 3]
Matrix  →  2D Tensor  →  [[1,2],[3,4]]
3D+     →  nD Tensor  →  stacked matrices (images, video, batches)
```

**Math notation:**

A scalar is written as x ∈ ℝ

A vector is written as **x** ∈ ℝⁿ

A matrix is written as **X** ∈ ℝᵐˣⁿ

A 3D tensor is written as 𝒯 ∈ ℝᵈˣᵐˣⁿ

---

### 1.2 CREATING TENSORS — SYNTAX TABLE

```
Operation                          Syntax
-----------------------------------------------------------------
From Python list                   torch.tensor([1, 2, 3])
Zeros tensor                       torch.zeros(3, 4)
Ones tensor                        torch.ones(3, 4)
Random uniform [0,1)               torch.rand(3, 4)
Random normal (μ=0, σ=1)           torch.randn(3, 4)
Range of values                    torch.arange(0, 10, 2)
Linspace                           torch.linspace(0, 1, steps=5)
Identity matrix                    torch.eye(3)
Like another tensor (zeros)        torch.zeros_like(existing_tensor)
Like another tensor (ones)         torch.ones_like(existing_tensor)
Empty (uninitialized)              torch.empty(2, 3)
-----------------------------------------------------------------
```

**Key to Remember:** torch.tensor() copies data. torch.as_tensor() shares memory when possible (efficient for NumPy arrays).

---

### 1.3 TENSOR ATTRIBUTES — THE BIG THREE

Every tensor has three fundamental properties you must always know:

**1. dtype** — what type of data is stored

```
torch.float32    (default for floats, most neural net weights)
torch.float64    (double precision)
torch.float16    (half precision, used in GPU training to save memory)
torch.int32
torch.int64      (default for integers)
torch.bool
```

**2. shape** — dimensions of the tensor

shape returns a torch.Size object. Think of it as a tuple describing each axis.

For a tensor of shape (32, 3, 224, 224):
- 32 = batch size
- 3 = color channels (RGB)
- 224 = height
- 224 = width

**3. device** — where the tensor lives

```
cpu    → normal RAM
cuda   → GPU memory (NVIDIA)
mps    → Apple Silicon GPU
```

**Syntax to check:**
```
t.dtype
t.shape      (or t.size())
t.device
t.ndim       (number of dimensions)
t.numel()    (total number of elements)
```

---

### 1.4 DEVICE MANAGEMENT — CRITICAL CONCEPT

```
device = "cuda" if torch.cuda.is_available() else "cpu"

tensor_on_gpu = tensor.to(device)
tensor_on_gpu = tensor.cuda()     (shorthand)
tensor_back   = tensor_on_gpu.cpu()
```

**Rule:** Two tensors must be on the SAME device to interact. Mixing cpu and cuda tensors throws a RuntimeError. This is one of the most common beginner errors.

---

### 1.5 TENSOR OPERATIONS

**Element-wise operations (math happens at each position independently):**

```
a + b         torch.add(a, b)
a - b         torch.sub(a, b)
a * b         torch.mul(a, b)       ← element-wise, NOT matrix multiply
a / b         torch.div(a, b)
a ** 2        torch.pow(a, 2)
torch.sqrt(a)
torch.exp(a)
torch.log(a)
```

**Matrix Multiplication — THE most important operation:**

Given **A** ∈ ℝᵐˣⁿ and **B** ∈ ℝⁿˣᵖ

Result **C** = **AB** ∈ ℝᵐˣᵖ

Where each element: C_ij = Σ(k=1 to n) A_ik × B_kj

```
torch.matmul(A, B)    ← preferred, works for batched too
A @ B                 ← same thing, Python operator
torch.mm(A, B)        ← only for 2D, no batching
```

**Aggregation operations:**

```
t.sum()
t.mean()
t.max()
t.min()
t.std()
t.argmax()    ← index of max value (very common in classification)
t.argmin()
```

Pass `dim=` to aggregate along a specific axis:

```
t.sum(dim=0)    ← collapse rows, result has shape of one row
t.sum(dim=1)    ← collapse columns, result has shape of one column
```

**Math of mean:** μ = (1/n) Σᵢ xᵢ

**Math of standard deviation:** σ = √( (1/n) Σᵢ (xᵢ - μ)² )

---

### 1.6 SHAPE MANIPULATION — VERY IMPORTANT

```
t.reshape(rows, cols)      ← returns new shape, may copy
t.view(rows, cols)         ← returns new shape, shares memory (must be contiguous)
t.squeeze()                ← removes all dimensions of size 1
t.unsqueeze(dim=0)         ← adds a dimension at position 0
t.permute(2, 0, 1)         ← reorders dimensions (common in image processing)
t.transpose(0, 1)          ← swaps two specific dimensions
t.flatten()                ← collapses to 1D
torch.cat([a,b], dim=0)    ← concatenate along existing dimension
torch.stack([a,b], dim=0)  ← stack, creates NEW dimension
```

**The -1 trick in reshape:**

You can use -1 as a wildcard and PyTorch infers the correct value.

```
t.reshape(32, -1)   ← if t has 32*N elements, second dim becomes N automatically
```

**Why permute matters:**

NumPy/PIL images are (H, W, C) — Height, Width, Channels
PyTorch expects (C, H, W) — Channels, Height, Width

So you always do: tensor.permute(2, 0, 1) to convert.

---

### 1.7 INDEXING AND SLICING

PyTorch indexing works like NumPy:

```
t[0]           first element / first row
t[-1]          last
t[0, 1]        row 0, col 1
t[:, 1]        all rows, column 1
t[0:3, :]      rows 0,1,2 all columns
t[t > 0.5]     boolean masking
```

---

### MEMORIZE TABLE — TENSOR CREATION CHEAT SHEET

```
Need                          Use
----------------------------------------------------
Specific values               torch.tensor([...])
All zeros                     torch.zeros(shape)
All ones                      torch.ones(shape)
Random floats 0–1             torch.rand(shape)
Standard normal               torch.randn(shape)
Integer sequence              torch.arange(start, end, step)
Evenly spaced floats          torch.linspace(start, end, n)
Same shape as tensor x        torch.zeros_like(x)
----------------------------------------------------
```

---

## HOUR 2 — AUTOGRAD AND THE NEURAL NETWORK CORE

---

### 2.1 WHAT IS AUTOGRAD?

**Definition:** Autograd is PyTorch's automatic differentiation engine. It records all operations on tensors that have requires_grad=True, builds a computation graph, and then computes gradients automatically using backpropagation.

This is the mathematical engine that makes training neural networks possible.

---

### 2.2 THE MATH BEHIND BACKPROPAGATION

To train a neural network, we need to minimize a loss function L with respect to parameters (weights) w.

We do this using **Gradient Descent:**

w_new = w_old - η × (∂L/∂w)

Where:
- η (eta) = learning rate, a small positive number like 0.01
- ∂L/∂w = gradient of loss with respect to weight

The gradient tells us: "if I increase w by a tiny amount, how much does L change?"

**Chain Rule (why backprop works):**

If L = f(g(w)), then:

∂L/∂w = (∂L/∂g) × (∂g/∂w)

This chain rule extends through all layers of the network. PyTorch tracks every operation and applies this automatically.

**Simple example:**

Let y = x² and L = y

∂L/∂x = ∂L/∂y × ∂y/∂x = 1 × 2x = 2x

If x = 3, gradient = 6.

In PyTorch:
```
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)    # prints 6.0  ← PyTorch computed 2x = 2*3 = 6
```

---

### 2.3 AUTOGRAD SYNTAX AND WORKFLOW

```
x = torch.tensor(value, requires_grad=True)

y = some_operations(x)   ← computation graph built here

y.backward()             ← gradients computed via chain rule

x.grad                   ← access the gradient ∂y/∂x
```

**Key concepts:**

requires_grad=True → "track gradients for this tensor"

grad_fn → every tensor produced by an operation has this, showing what created it

.backward() → triggers backpropagation from that tensor

.grad → stores the accumulated gradient

**Stopping gradient tracking:**

```
with torch.no_grad():
    prediction = model(x)    ← no graph built, faster, used during inference

tensor.detach()              ← returns tensor with no grad history
```

**Why stop tracking?** During evaluation/inference you don't need gradients. Disabling saves memory and computation.

---

### 2.4 THE TRAINING LOOP — MEMORIZE THIS STRUCTURE

```
for epoch in range(num_epochs):

    # 1. Forward pass — compute prediction
    y_pred = model(X)

    # 2. Compute loss
    loss = loss_fn(y_pred, y_true)

    # 3. Zero gradients (CRITICAL — gradients accumulate by default)
    optimizer.zero_grad()

    # 4. Backward pass — compute gradients
    loss.backward()

    # 5. Update weights
    optimizer.step()
```

**Why zero_grad() matters:**

PyTorch accumulates gradients (adds to existing .grad) instead of replacing them. If you forget zero_grad(), gradients from previous batches contaminate the current update. This is intentional design for certain advanced use cases (like RNNs over sequences) but a common bug for beginners.

---

### 2.5 nn.Module — BUILDING NEURAL NETWORKS

**Definition:** nn.Module is the base class for all neural network models in PyTorch. Every custom model inherits from it.

**Two things you must implement:**

1. __init__ — define layers
2. forward — define how data flows through layers

```
import torch.nn as nn

class MyNet(nn.Module):

    def __init__(self):
        super().__init__()              ← ALWAYS call this first
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = MyNet()
```

---

### 2.6 KEY LAYERS — DEFINITION AND MATH

**nn.Linear(in_features, out_features)**

Performs: output = x × Wᵀ + b

Where W ∈ ℝ^(out × in) is the weight matrix and b ∈ ℝ^out is the bias vector.

This is a fully connected / dense layer.

**nn.ReLU()**

ReLU (Rectified Linear Unit): f(x) = max(0, x)

Derivative: f'(x) = 1 if x > 0, else 0

Why use it? Introduces non-linearity. Without activation functions, stacking linear layers = one linear layer (useless for complex patterns).

**nn.Sigmoid()**

f(x) = 1 / (1 + e^(-x))

Maps any value to range (0, 1). Used in binary classification output layers.

Derivative: f'(x) = f(x)(1 - f(x))

**nn.Tanh()**

f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Maps to range (-1, 1). Often preferred over sigmoid in hidden layers.

**nn.Softmax(dim=1)**

f(xᵢ) = e^(xᵢ) / Σⱼ e^(xⱼ)

Converts raw scores (logits) to probabilities that sum to 1. Used in multiclass classification output.

---

### 2.7 LOSS FUNCTIONS

**Mean Squared Error (regression):**

MSE = (1/n) Σᵢ (ŷᵢ - yᵢ)²

```
nn.MSELoss()
```

**Binary Cross Entropy (binary classification):**

BCE = -(1/n) Σᵢ [ yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ) ]

```
nn.BCELoss()           ← expects sigmoid output
nn.BCEWithLogitsLoss() ← applies sigmoid internally, numerically stable, PREFERRED
```

**Cross Entropy Loss (multiclass classification):**

CE = -(1/n) Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)

```
nn.CrossEntropyLoss()  ← expects raw logits (no softmax before), applies log-softmax internally
```

**Key to remember:** nn.CrossEntropyLoss = LogSoftmax + NLLLoss combined. Never apply softmax before it, or you'll double-apply and get wrong results.

---

### 2.8 OPTIMIZERS

**Definition:** An optimizer updates model parameters using computed gradients.

```
SGD (Stochastic Gradient Descent):
w = w - η × ∇L

torch.optim.SGD(model.parameters(), lr=0.01)
torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Adam (Adaptive Moment Estimation):**

Adam maintains per-parameter adaptive learning rates. It tracks:
- m_t = β₁ m_{t-1} + (1-β₁) g_t        (first moment, like momentum)
- v_t = β₂ v_{t-1} + (1-β₂) g_t²       (second moment, like RMSProp)
- w = w - η × m̂_t / (√v̂_t + ε)

Default: β₁=0.9, β₂=0.999, ε=1e-8

```
torch.optim.Adam(model.parameters(), lr=0.001)
```

Adam is the default choice for most deep learning tasks. SGD with momentum sometimes converges to better solutions for vision tasks.

---

### MEMORIZE TABLE — LOSS FUNCTION SELECTION

```
Task                         Loss Function                    Output Activation
-------------------------------------------------------------------------------
Regression                   nn.MSELoss()                     None (linear)
Binary classification        nn.BCEWithLogitsLoss()           None (raw logit)
Multiclass classification    nn.CrossEntropyLoss()            None (raw logits)
-------------------------------------------------------------------------------
```

---

## HOUR 3 — COMPLETE WORKFLOW, DATASETS, AND ADVANCED CONCEPTS

---

### 3.1 DATASETS AND DATALOADERS

**Definition:** Dataset is an abstract class representing your data. DataLoader wraps a Dataset and handles batching, shuffling, and parallel loading.

**Custom Dataset:**

```
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)                 ← tells DataLoader total size

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]   ← returns one sample

dataset = MyDataset(X_tensor, y_tensor)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)
```

**DataLoader parameters:**

```
batch_size    how many samples per batch
shuffle       True for training, False for val/test
num_workers   parallel data loading threads (0 = main thread only)
drop_last     drop final batch if smaller than batch_size
```

**Iterating:**

```
for X_batch, y_batch in loader:
    # X_batch shape: (batch_size, features)
    # y_batch shape: (batch_size,)
    ...
```

---

### 3.2 MODEL MODES — CRITICAL DISTINCTION

```
model.train()   ← enables dropout, batch norm in training mode
model.eval()    ← disables dropout, uses running stats for batch norm
```

Always switch modes at the right time:

```
# Training
model.train()
for X, y in train_loader:
    ...

# Evaluation
model.eval()
with torch.no_grad():
    for X, y in val_loader:
        ...
```

---

### 3.3 SAVING AND LOADING MODELS

**Saving (recommended way — save state_dict only):**

```
torch.save(model.state_dict(), "model.pth")
```

**Loading:**

```
model = MyNet()                            ← create architecture first
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

state_dict is a Python dictionary mapping layer names to their parameter tensors. This is preferred over saving the whole model because it's more portable.

---

### 3.4 COMPLETE TRAINING PIPELINE — REFERENCE CODE

```
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Data ---
X = torch.randn(1000, 20)
y = torch.randint(0, 3, (1000,))

dataset    = TensorDataset(X, y)
train_size = 800
val_size   = 200
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False)

# --- Model ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = Net().to(device)

# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training Loop ---
for epoch in range(10):
    model.train()
    total_loss = 0
    for X_b, y_b in train_dl:
        X_b, y_b = X_b.to(device), y_b.to(device)
        pred = model(X_b)
        loss = criterion(pred, y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = 0
    with torch.no_grad():
        for X_b, y_b in val_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred    = model(X_b)
            correct += (pred.argmax(1) == y_b).sum().item()
    acc = correct / val_size
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_dl):.4f} | Val Acc: {acc:.4f}")
```

---

### 3.5 nn.Sequential — SHORTCUT FOR SIMPLE ARCHITECTURES

```
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

nn.Sequential chains layers and the forward pass just passes input through each one in order. Great for linear stacks. Use full nn.Module when you need skip connections, multiple inputs/outputs, or branching.

---

### 3.6 DROPOUT AND BATCH NORMALIZATION

**Dropout (regularization):**

During training, randomly zeros out neurons with probability p. This prevents co-adaptation and overfitting.

```
nn.Dropout(p=0.5)    ← 50% of neurons dropped during training
```

During eval(), dropout is automatically disabled (all neurons active, outputs scaled).

**Batch Normalization:**

For a mini-batch, normalizes activations:

x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)

Then scales and shifts: y = γx̂ + β (γ and β are learned parameters)

Benefits: faster training, allows higher learning rates, mild regularization effect.

```
nn.BatchNorm1d(num_features)    ← for fully connected layers
nn.BatchNorm2d(num_channels)    ← for convolutional layers
```

---

### 3.7 CONVOLUTIONAL LAYERS (CNN BASICS)

**nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)**

A convolution slides a kernel (small filter) over the input, computing dot products.

Output spatial dimensions:

H_out = ⌊(H_in + 2×padding - kernel_size) / stride⌋ + 1

W_out = ⌊(W_in + 2×padding - kernel_size) / stride⌋ + 1

**Example:**

Input: (batch=1, channels=3, H=28, W=28)
Conv2d(3, 16, kernel_size=3, padding=1)
Output: (1, 16, 28, 28)  ← same spatial size because padding=1

**MaxPool2d:**

```
nn.MaxPool2d(kernel_size=2, stride=2)   ← halves spatial dimensions
```

---

### 3.8 TIPS AND TRICKS TO IMPLEMENT

**Tip 1 — Always check tensor shapes:**
When debugging, print tensor shapes at each step. More than 90% of shape errors are caught this way. Use t.shape liberally.

**Tip 2 — Use loss.item():**
loss is a tensor on GPU. loss.item() extracts the scalar Python float. Use this for logging, not loss itself.

**Tip 3 — Normalize your data:**
Neural networks train better when inputs are roughly in range (-1, 1) or (0, 1). Use (x - mean) / std.

**Tip 4 — Start with Adam, lr=1e-3:**
Default starting point for most tasks. Tune from here.

**Tip 5 — Learning rate is your most important hyperparameter:**
Too high → loss explodes. Too low → training is slow and may get stuck. Try: 1e-2, 1e-3, 1e-4.

**Tip 6 — Use torch.no_grad() everywhere during inference:**
Not just for correctness — it significantly speeds up computation and reduces memory use.

**Tip 7 — reproducibility:**
```
torch.manual_seed(42)
```
Set this at the start of every experiment.

**Tip 8 — model.parameters() vs model.state_dict():**
model.parameters() → iterator used by optimizer
model.state_dict() → dictionary used for saving/loading

**Tip 9 — Never put softmax before CrossEntropyLoss:**
nn.CrossEntropyLoss does it internally. Double-applying softmax gives silent wrong results, not an error.

**Tip 10 — For classification, use argmax on output:**
```
predictions = model(X).argmax(dim=1)
accuracy = (predictions == y).float().mean()
```

---

## PRACTICE QUESTION SET

---

### SECTION A — CONCEPTUAL (Write answers in your own words)

Q1. Explain the difference between a static computation graph and a dynamic computation graph. Why does PyTorch's approach aid debugging?

Q2. What are the three core attributes every PyTorch tensor has? Explain what each one represents.

Q3. Explain why optimizer.zero_grad() must be called before loss.backward() in the training loop. What goes wrong if you skip it?

Q4. Why should you never apply softmax manually before passing outputs to nn.CrossEntropyLoss()?

Q5. What is the difference between model.train() and model.eval()? Name two layers that behave differently depending on this mode.

Q6. Explain the chain rule in the context of backpropagation. If L = f(g(h(x))), write out ∂L/∂x.

Q7. What does requires_grad=True tell PyTorch? What is the computation graph and when is it built?

Q8. What is the difference between torch.mm(), torch.matmul(), and the @ operator?

Q9. Why is the state_dict preferred over saving the entire model with torch.save(model)?

Q10. Explain what happens in a DataLoader when shuffle=True. Why is this important for training?

---

### SECTION B — SHAPE TRACING (Critical skill — trace without running)

Q11. Given t = torch.randn(4, 3, 28, 28), what is the shape after:
- t.view(4, -1)
- t.permute(0, 2, 3, 1)
- t.squeeze()
- t[:, 0, :, :]

Q12. Given a = torch.randn(32, 64) and b = torch.randn(64, 128), what is the shape of a @ b?

Q13. Given t = torch.randn(8, 16), what does t.unsqueeze(0).unsqueeze(2).shape return?

Q14. Given t = torch.randn(3, 1, 5), what does t.squeeze().shape return?

Q15. You have a tensor of shape (100,). You want shape (100, 1). What do you call?

---

### SECTION C — CODE DEBUGGING (Find the bug)

Q16. What is wrong with this code?
```
x = torch.randn(10, 5)
y = torch.randn(10, 5)
# User says: "I want matrix multiplication"
result = x * y
```

Q17. Find the bug:
```
model = MyNet()
x = torch.randn(32, 784).to("cuda")
output = model(x)
```

Q18. Find the bug:
```
for epoch in range(10):
    for X, y in loader:
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
```

Q19. Find the bug:
```
logits = model(x)
probs  = nn.Softmax(dim=1)(logits)
loss   = nn.CrossEntropyLoss()(probs, y)
```

Q20. Find the bug:
```
model.eval()
for X, y in val_loader:
    pred = model(X)
    loss = criterion(pred, y)
    acc  = (pred.argmax(1) == y).float().mean()
```

---

### SECTION D — IMPLEMENTATION (Write the code)

Q21. Create a tensor of shape (5, 5) filled with random normal values. Compute its mean and standard deviation along dimension 0. What are the shapes of the results?

Q22. Write a function that takes a numpy array and returns a float32 CUDA tensor (or CPU tensor if no GPU available).

Q23. Build a neural network using nn.Module that:
- Takes input of size 100
- Has two hidden layers of size 256 and 128 with ReLU
- Has dropout of 0.4 after each hidden layer
- Outputs 10 classes (raw logits)

Q24. Write a complete single-epoch training step as a function that takes model, loader, criterion, optimizer, device and returns average loss.

Q25. Write code to:
- Train for 5 epochs
- Track and print train loss and validation accuracy each epoch
- Save the model at the end

---

### SECTION E — MATH AND CONCEPTS

Q26. Given x = torch.tensor(4.0, requires_grad=True), compute y = 3x³ + 2x + 1. What is dy/dx at x=4? Verify by computing manually and then describe what PyTorch would give.

Q27. A Linear layer has in_features=512 and out_features=256. How many trainable parameters does it have (including bias)?

Q28. A Conv2d layer with in_channels=3, out_channels=64, kernel_size=3. How many parameters does it have? (Hint: include bias)

Q29. An image tensor has shape (H=256, W=256, C=3) as loaded by PIL. What single operation converts it to PyTorch's expected format?

Q30. Your model outputs shape (32, 10) logits for a batch of 32 samples across 10 classes. Write the lines to get predicted class labels and compute accuracy against true labels y of shape (32,).

---

## SOLUTIONS

---

### SECTION A — SOLUTIONS

A1. Static graphs are compiled before execution (TensorFlow 1.x style) — the full computation path is fixed. Dynamic graphs are built as code executes, making them identical to regular Python. Dynamic approach means you can use print() for debugging, standard Python if/else for control flow, and a Python debugger (pdb) directly on the model. PyTorch's dynamic graph makes debugging feel natural.

A2. dtype (data type — what kind of values are stored), shape (dimensions — size along each axis), device (where the tensor lives — cpu or cuda). These three define everything about where and how a tensor exists.

A3. Gradients accumulate (add to .grad) in PyTorch. Skipping zero_grad() means the current batch's gradients are added on top of all previous batches, making the gradient increasingly wrong. The optimizer then makes an incorrect update.

A4. nn.CrossEntropyLoss internally applies log-softmax. If you apply softmax first and then pass to CrossEntropyLoss, it applies softmax again on an already-normalized distribution, producing mathematically incorrect loss values without throwing any error.

A5. model.train() sets the model to training mode. model.eval() sets it to evaluation mode. Dropout: active during train (randomly zeros neurons), disabled during eval. BatchNorm: during train uses batch statistics (μ and σ from current batch), during eval uses running averages accumulated over training.

A6. Chain rule: ∂L/∂x = (∂L/∂f) × (∂f/∂g) × (∂g/∂h) × (∂h/∂x). Each layer's gradient multiplies through the chain, going backwards from loss to input.

A7. requires_grad=True tells PyTorch to track all operations on that tensor in a computation graph. The computation graph is a directed acyclic graph where nodes are tensors and edges are operations. It is built dynamically during the forward pass as operations execute. .backward() traverses it in reverse.

A8. torch.mm() works only for 2D matrix multiplication, no batching. torch.matmul() supports broadcasting and batched matrix multiplication (e.g., (batch, m, n) @ (batch, n, p)). The @ operator calls torch.matmul(). For 2D only, they're identical. For 3D+, use matmul or @.

A9. state_dict only saves the parameter tensors as a dictionary — it is architecture-agnostic and smaller. Saving the whole model with torch.save(model) pickles the class itself, which can break if you refactor the code, rename files, or load in a different codebase. state_dict + reconstruct architecture is more robust and portable.

A10. With shuffle=True, the DataLoader randomly permutes the dataset order at the start of each epoch. This ensures the model doesn't learn any artificial order in the data, every batch has a different composition each epoch, and gradient updates are more varied and less correlated, which helps convergence.

---

### SECTION B — SOLUTIONS

B11.
- t.view(4, -1) → shape (4, 2352) because 3×28×28 = 2352
- t.permute(0, 2, 3, 1) → shape (4, 28, 28, 3)
- t.squeeze() → shape (4, 3, 28, 28) unchanged, no size-1 dims
- t[:, 0, :, :] → shape (4, 28, 28)

B12. (32, 64) @ (64, 128) → (32, 128). Inner dimensions must match; outer dimensions form the result.

B13. Start: (8, 16). After unsqueeze(0): (1, 8, 16). After unsqueeze(2): (1, 8, 1, 16). Shape is torch.Size([1, 8, 1, 16]).

B14. Shape (3, 1, 5). squeeze() removes the size-1 dimension (the middle one). Result: (3, 5).

B15. t.unsqueeze(1) or t.reshape(100, 1) or t.view(100, 1).

---

### SECTION C — SOLUTIONS

C16. x * y performs element-wise multiplication, not matrix multiplication. For matrix multiplication you need x @ y or torch.matmul(x, y). Note: element-wise requires same shape, matrix multiplication requires x's last dim = y's first dim.

C17. The model is on CPU (default) but x is on CUDA. You must call model.to("cuda") or model.cuda() before passing GPU tensors to it.

C18. optimizer.zero_grad() is missing before loss.backward(). Gradients accumulate across iterations, corrupting updates from the second batch onward.

C19. Softmax is applied manually to logits before passing to nn.CrossEntropyLoss. CrossEntropyLoss internally applies log-softmax, so this double-applies it. Remove the softmax line and pass raw logits directly to the loss.

C20. torch.no_grad() is missing. During evaluation, gradients are being tracked unnecessarily, wasting memory and computation. Wrap the eval loop in with torch.no_grad():.

---

### SECTION D — SOLUTIONS

D21.
```
t = torch.randn(5, 5)
mean = t.mean(dim=0)     # shape: (5,) — mean of each column
std  = t.std(dim=0)      # shape: (5,)
```

D22.
```
def to_tensor(numpy_arr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(numpy_arr, dtype=torch.float32).to(device)
```

D23.
```
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)
```

D24.
```
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
```

D25.
```
for epoch in range(5):
    train_loss = train_epoch(model, train_dl, criterion, optimizer, device)

    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.to(device), y.to(device)
            pred     = model(X)
            correct += (pred.argmax(1) == y).sum().item()
            total   += y.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "model.pth")
```

---

### SECTION E — SOLUTIONS

E26. y = 3x³ + 2x + 1

dy/dx = 9x² + 2

At x = 4: 9(16) + 2 = 144 + 2 = 146

PyTorch would compute x.grad = tensor(146.) after y.backward().

E27. Parameters = (in × out) + out = (512 × 256) + 256 = 131072 + 256 = 131,328

E28. Each filter has (in_channels × kernel_size × kernel_size) weights + 1 bias

= (3 × 3 × 3) + 1 per filter = 27 + 1 = 28 per filter

64 filters → 64 × 28 = 1,792 total parameters.

E29. tensor.permute(2, 0, 1) — converts from (H, W, C) to (C, H, W).

E30.
```
predictions = output.argmax(dim=1)            # shape (32,)
accuracy    = (predictions == y).float().mean().item()
```

---

## FINAL QUICK REFERENCE — WHAT TO LOCK IN YOUR HEAD

```
TENSOR CREATION:          torch.tensor(), torch.zeros(), torch.rand(), torch.randn()
TENSOR MANIPULATION:      reshape, view, permute, squeeze, unsqueeze, cat, stack
MATRIX MULTIPLY:          @ or torch.matmul() — never * for matmul
GRADIENT FLOW:            requires_grad → forward → .backward() → .grad
TRAINING ORDER:           forward → loss → zero_grad → backward → step
LOSS SELECTION:           regression=MSE, binary=BCEWithLogits, multi=CrossEntropy
OPTIMIZER DEFAULT:        Adam, lr=1e-3
MODEL MODES:              .train() for training, .eval() for inference
NO_GRAD:                  always use during eval/inference
DEVICE RULE:              model and data must be on same device
SAVING:                   save state_dict, not whole model
```

---

Good luck locking in — you've got everything you need right here. The practice questions at the end are designed to cover the exact gaps that trip people up in interviews and implementation. Work through them in order: conceptual first to build the why, then shape tracing (do it on paper), then debugging, then write the code yourself without looking. That 3-hour arc will hardwire this foundation.