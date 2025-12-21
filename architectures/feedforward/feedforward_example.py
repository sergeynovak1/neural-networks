"""
Пример многослойного перцептрона (MLP) для задачи классификации
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


# Определение архитектуры MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        # Создание списка слоев
        layers = []
        prev_size = input_size
        
        # Добавляем скрытые слои
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Dropout для регуляризации
            prev_size = hidden_size
        
        # Выходной слой
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))  # Softmax для многоклассовой классификации
        
        # Объединяем все слои в последовательность
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def generate_data():
    """Генерация синтетических данных для классификации"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def train_model():
    # Генерация данных
    X_train, X_test, y_train, y_test = generate_data()
    
    # Преобразование данных в тензоры PyTorch
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Параметры модели
    input_size = X_train.shape[1]
    hidden_sizes = [64, 32]  # Архитектура скрытых слоев
    output_size = len(torch.unique(y_train))
    
    # Инициализация модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size, hidden_sizes, output_size).to(device)
    
    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Конвертация данных в тензоры и перемещение на устройство
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)
    
    # Цикл обучения
    num_epochs = 100
    batch_size = 32
    
    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        total_loss = 0
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Вывод статистики каждые 20 эпох
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/(X_train.size()[0]/batch_size):.4f}')
    
    # Тестирование модели
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f'Accuracy on test set: {100 * accuracy:.2f}%')


if __name__ == "__main__":
    train_model()