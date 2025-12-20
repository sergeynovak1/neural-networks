"""
Пример генеративно-состязательной сети (GAN) для генерации изображений MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Генератор
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_dim=28*28):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh()  # Значения в диапазоне [-1, 1]
        )
    
    def forward(self, z):
        return self.model(z)


# Дискриминатор
class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Вероятность, что изображение настоящее
        )
    
    def forward(self, img):
        return self.model(img)


def train_gan():
    # Подготовка данных MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Нормализация в диапазон [-1, 1]
    ])
    
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    
    # Параметры
    latent_dim = 100
    img_dim = 28*28
    
    # Инициализация моделей
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim, img_dim).to(device)
    discriminator = Discriminator(img_dim).to(device)
    
    # Определение функции потерь и оптимизаторов
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Метки для реальных и поддельных изображений
    real_label = 1.
    fake_label = 0.
    
    # Цикл обучения
    num_epochs = 50
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(trainloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.view(batch_size, -1).to(device)
            
            # ===========================
            # Обучение дискриминатора
            # ===========================
            optimizer_D.zero_grad()
            
            # Реальные изображения
            labels_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output_real = discriminator(real_imgs).view(-1)
            loss_D_real = criterion(output_real, labels_real)
            
            # Поддельные изображения
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(noise)
            labels_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
            output_fake = discriminator(fake_imgs.detach()).view(-1)
            loss_D_fake = criterion(output_fake, labels_fake)
            
            # Общая потеря дискриминатора
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()
            
            # ===========================
            # Обучение генератора
            # ===========================
            optimizer_G.zero_grad()
            
            # Поддельные изображения (повторно, но без detach)
            output_fake = discriminator(fake_imgs).view(-1)
            loss_G = criterion(output_fake, labels_real)  # Хотим, чтобы дискриминатор считал поддельные изображения реальными
            loss_G.backward()
            optimizer_G.step()
            
            # Вывод статистики каждые 200 батчей
            if i % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch {i}/{len(trainloader)}, '
                      f'Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}')
        
        # Визуализация генерируемых изображений каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_noise = torch.randn(16, latent_dim, device=device)
                generated_imgs = generator(sample_noise).cpu().view(-1, 1, 28, 28)
                
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for j in range(16):
                    ax = axes[j//4, j%4]
                    ax.imshow(generated_imgs[j].squeeze(), cmap='gray')
                    ax.axis('off')
                
                plt.suptitle(f'Generated Images at Epoch {epoch+1}')
                plt.tight_layout()
                plt.show()
    
    # Сохранение моделей
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print("Модели сохранены")


if __name__ == "__main__":
    train_gan()