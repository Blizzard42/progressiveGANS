import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

# ------------------ CONFIG ------------------
image_size = 32
nz = 100  # Latent vector size
gen_hidden = 64
disc_hidden = 64
batch_size = 128
epochs = 100
early_stop_patience = 2
early_stop_threshold = 0.01
k = 1000  # Number of generator/discriminator pairs to train
save_dir = "generated_data"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ DATASET ------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(cifar, batch_size=batch_size, shuffle=True, num_workers=2)

# ------------------ MODELS ------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nz, gen_hidden * 8),
            nn.BatchNorm1d(gen_hidden * 8),
            nn.ReLU(True),
            nn.Linear(gen_hidden * 8, image_size * image_size * 3),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, 3, image_size, image_size)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size * 3, disc_hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(disc_hidden * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ------------------ TRAINING UTILS ------------------
def early_stopping(losses, threshold=early_stop_threshold, patience=early_stop_patience):
    if len(losses) < patience:
        return False
    recent = losses[-patience:]
    return max(recent) - min(recent) < threshold

def save_generated_images(generator, stage, num_samples=25000):
    generator.eval()
    z = torch.randn(num_samples, nz).to(device)
    with torch.no_grad():
        samples = generator(z).cpu()
    torch.save(samples, os.path.join(save_dir, f"gen_outputs_stage_{stage}.pt"))

def load_generated_images_range(start_stage, end_stage):
    fakes = []
    for s in range(start_stage, end_stage + 1):
        path = os.path.join(save_dir, f"gen_outputs_stage_{s}.pt")
        if os.path.exists(path):
            fakes.append(torch.load(path))
    return fakes

def visualize_images(generator, num_images=16):
    generator.eval()
    z = torch.randn(num_images, nz).to(device)
    with torch.no_grad():
        samples = generator(z).cpu() * 0.5 + 0.5
    grid = torchvision.utils.make_grid(samples, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.show()

def train_discriminator(discriminator, real_loader, fake_collections, criterion, optimizer):
    discriminator.train()
    losses = []

    real_images = torch.stack([x for x, _ in real_loader.dataset])
    num_real = len(real_images)

    latest_fake = fake_collections[0][:int(0.5 * num_real)]
    recent_fakes = [f[:int(0.1 * num_real)] for f in fake_collections[1:]]

    fake_images = torch.cat([latest_fake] + recent_fakes)

    fake_dataset = TensorDataset(fake_images, torch.zeros(len(fake_images)))
    real_dataset = TensorDataset(real_images, torch.ones(len(real_images)))

    combined_dataset = ConcatDataset([real_dataset, fake_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    print(f"\nDiscriminator Training with {len(fake_images)} fake and {len(real_images)} real images")
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress = tqdm(combined_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = discriminator(inputs).squeeze()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(combined_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: D Loss = {avg_loss:.4f}")
        if early_stopping(losses):
            print("Convergence Detected: exiting discriminator training early")
            break

def train_generator(generator, frozen_discriminator, criterion, optimizer):
    generator.train()
    losses = []

    print("\nGenerator Training")
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress = tqdm(range(len(dataloader)), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for _ in progress:
            z = torch.randn(batch_size, nz).to(device)
            fake_images = generator(z)

            frozen_discriminator.eval()
            with torch.no_grad():
                predictions = frozen_discriminator(fake_images).squeeze()

            labels = torch.ones_like(predictions)

            optimizer.zero_grad()
            predictions = frozen_discriminator(fake_images).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: G Loss = {avg_loss:.4f}")
        if early_stopping(losses):
            print("Early stopping generator.")
            break

# ------------------ MAIN TRAINING LOOP ------------------
for stage in range(1, k+1):
    print(f"\n============================")
    print(f"      Stage {stage}/{k}")
    print(f"============================")

    D = Discriminator().to(device)
    G = Generator().to(device)

    d_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    if stage == 1:
        fake_data = torch.randn(50000, 3, image_size, image_size).clamp(-1, 1)
        fake_collections = [fake_data]
    else:
        fake_collections = load_generated_images_range(max(1, stage - 5), stage - 1)[::-1]

    print("Training discriminator...")
    train_discriminator(D, dataloader, fake_collections, criterion, d_optim)

    print("Training generator...")
    train_generator(G, D, criterion, g_optim)

    # print("Visualizing current generator output...")
    # visualize_images(G)

    print("Saving generated images...")
    save_generated_images(G, stage)

print("\nTraining complete!")
