import torch
x = torch.tensor([1.,2.,3.,4.])
yt = torch.tensor([2.,4.,6.,8.])
w =torch.tensor(0.0,requires_grad=True)
lr = 0.1
epochs = 10
for epoch in range(epochs):
    yp = (w * x)
    loss = ((yp -yt)**2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
    print(f"epoch{epoch}: w = {w.item():.4f}, loss = {loss.item():.4f}")