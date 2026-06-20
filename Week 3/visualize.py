import torch
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms

from model import CNNModel

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNNModel().to(device)

model.load_state_dict(
    torch.load(
        "cnn_model.pth",
        map_location=device
    )
)

model.eval()

activations = {}

def activation_hook(name):

    def hook(module, inp, output):
        activations[name] = output.detach()

    return hook

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))
])

dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

image, _ = dataset[0]
image = image.unsqueeze(0).to(device)


# TODO:
# Register hooks
model.conv_block1.register_forward_hook(
    activation_hook("conv_block1")
)

model.conv_block2.register_forward_hook(
    activation_hook("conv_block2"))

# TODO:
# Run inference on a sample image
with torch.no_grad():
    _ = model(image)

# Plot feature maps
for layer_name in activations:
    # TODO: define feature maps and plot
    feature_maps = activations[layer_name][0]

    num_maps = feature_maps.shape[0]

    plt.figure(figsize=(12, 6))

    for i in range(num_maps):

        plt.subplot(2, (num_maps + 1)//2, i + 1)

        plt.imshow(
            feature_maps[i].cpu().numpy(),
            cmap="gray"
        )

        plt.axis("off")

    plt.suptitle(layer_name)

    plt.show()