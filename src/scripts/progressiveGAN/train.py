import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
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
early_stop_patience = 5
early_stop_threshold = 0.01
k = 3  # Number of generator/discriminator pairs to train
save_dir = "generated_data"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

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

def save_generated_images(generator, stage, num_samples=5000):
    generator.eval()
    z = torch.randn(num_samples, nz).to(device)
    with torch.no_grad():
        samples = generator(z).cpu()
    torch.save(samples, os.path.join(save_dir, f"gen_outputs_stage_{stage}.pt"))

def load_generated_images(upto_stage):
    all_fakes = []
    for s in range(1, upto_stage+1):
        fakes = torch.load(os.path.join(save_dir, f"gen_outputs_stage_{s}.pt"))
        all_fakes.append(fakes)
    return torch.cat(all_fakes)

def train_discriminator(discriminator, real_loader, fake_images, criterion, optimizer):
    discriminator.train()
    losses = []
    fake_dataset = TensorDataset(fake_images, torch.zeros(len(fake_images)))
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for real_batch, _ in real_loader:
            real_batch = real_batch.to(device)
            b_size = real_batch.size(0)
            fake_batch, _ = next(iter(fake_loader))
            fake_batch = fake_batch[:b_size].to(device)

            inputs = torch.cat([real_batch, fake_batch])
            labels = torch.cat([torch.ones(b_size), torch.zeros(b_size)]).to(device)

            optimizer.zero_grad()
            output = discriminator(inputs).squeeze()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(real_loader))
        if early_stopping(losses):
            print("Convergence Detected: exiting discriminator training early")
            break

def train_generator(generator, frozen_discriminator, criterion, optimizer):
    generator.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(len(dataloader)):
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

        losses.append(epoch_loss / len(dataloader))
        if early_stopping(losses):
            break

# ------------------ MAIN TRAINING LOOP ------------------
for stage in range(1, k+1):
    print(f"\n--- Stage {stage} ---")

    # Load previous generations for fake data
    if stage == 1:
        fake_data = torch.empty(0, 3, image_size, image_size)
    else:
        fake_data = load_generated_images(stage-1)

    D = Discriminator().to(device)
    G = Generator().to(device)

    d_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    print("Training discriminator...")
    train_discriminator(D, dataloader, fake_data, criterion, d_optim)

    print("Training generator...")
    train_generator(G, D, criterion, g_optim)

    print("Saving generated images...")
    save_generated_images(G, stage)

print("\nTraining complete!")
