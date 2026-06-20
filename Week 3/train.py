import torch
import torch.nn as nn
import torch.optim as optim

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import CNNModel

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5))
    # TODO: Normalize
])

train_dataset = datasets.CIFAR10(
    root="../data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):

    model.train()
    running_loss=0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # TODO: calculate backprop, gradient update and running_loss 

    print(
        f"Epoch [{epoch+1}/5] "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

torch.save(model.state_dict(), "cnn_model.pth")
print("Training complete.")

