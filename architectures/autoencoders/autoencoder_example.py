"""
Пример автоэнкодера для снижения размерности и восстановления изображений MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Определение архитектуры автоэнкодера
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim=64):
        super(Autoencoder, self).__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, encoding_dim),
            nn.ReLU(True)
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            nn.Sigmoid()  # Значения пикселей в диапазоне [0,1]
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        """Метод для получения кодирования"""
        return self.encoder(x)


def train_autoencoder():
    # Подготовка данных MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Преобразование в вектор
    ])
    
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)
    
    # Инициализация модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(encoding_dim=64).to(device)
    
    # Определение функции потерь и оптимизатора
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Цикл обучения
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in trainloader:
            inputs, _ = data
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Вывод статистики каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(trainloader):.4f}')
    
    # Визуализация результатов
    model.eval()
    with torch.no_grad():
        # Получаем несколько тестовых изображений
        dataiter = iter(testloader)
        images, _ = next(dataiter)
        images = images.to(device)
        
        # Получаем восстановленные изображения
        reconstructed = model(images)
        
        # Преобразуем тензоры обратно в изображения
        original_images = images.cpu().view(-1, 28, 28)
        reconstructed_images = reconstructed.cpu().view(-1, 28, 28)
        
        # Отображаем оригинальные и восстановленные изображения
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        for i in range(10):
            # Оригинальные изображения
            axes[0, i].imshow(original_images[i], cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Восстановленные изображения
            axes[1, i].imshow(reconstructed_images[i], cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Демонстрация снижения размерности
        encoded_sample = model.encode(images[:5])
        print(f"Original dimension: {images[0].shape}")
        print(f"Encoded dimension: {encoded_sample[0].shape}")


if __name__ == "__main__":
    train_autoencoder()