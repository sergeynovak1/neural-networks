"""
Пример рекуррентной нейронной сети (RNN) для анализа тональности текста
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


# Простая реализация RNN
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1):
        super(SimpleRNN, self).__init__()
        
        # Слой эмбеддинга
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN слой
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
        
        # Полносвязный слой для классификации
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Функция активации
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, sequence_length, embedding_dim)
        
        rnn_out, hidden = self.rnn(embedded)
        # rnn_out shape: (batch_size, sequence_length, hidden_dim)
        # hidden shape: (n_layers, batch_size, hidden_dim)
        
        # Используем последнее скрытое состояние
        last_hidden = hidden[-1]
        # last_hidden shape: (batch_size, hidden_dim)
        
        output = self.fc(last_hidden)
        return self.sigmoid(output)


# Пример датасета для анализа тональности
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Преобразование текста в последовательность индексов
        indices = [self.vocab.get(word, 0) for word in text.split()]
        # Паддинг или обрезка до фиксированной длины
        max_len = 50
        if len(indices) < max_len:
            indices.extend([0] * (max_len - len(indices)))
        else:
            indices = indices[:max_len]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)


def create_vocab(texts):
    """Создание словаря из текстов"""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def train_model():
    # Пример данных (в реальном проекте здесь будет загрузка настоящего датасета)
    train_texts = [
        "это отличный фильм я люблю его",
        "ужасный фильм не стоит смотреть",
        "хороший сюжет и актерская игра",
        "скучно и неинтересно",
        "прекрасная история с хорошим концом",
        "плохая режиссура и сценарий"
    ]
    train_labels = [1, 0, 1, 0, 1, 0]  # 1 - положительная тональность, 0 - отрицательная
    
    # Создание словаря
    vocab = create_vocab(train_texts)
    vocab_size = len(vocab)
    
    # Создание датасета и загрузчика
    dataset = SentimentDataset(train_texts, train_labels, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Параметры модели
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 1
    n_layers = 1
    
    # Инициализация модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers).to(device)
    
    # Определение функции потерь и оптимизатора
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Цикл обучения
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    # Тестирование на новых примерах
    model.eval()
    test_texts = [
        "это великолепный фильм",
        "очень скучный и длинный"
    ]
    
    with torch.no_grad():
        for text in test_texts:
            indices = [vocab.get(word, 0) for word in text.split()]
            max_len = 50
            if len(indices) < max_len:
                indices.extend([0] * (max_len - len(indices)))
            else:
                indices = indices[:max_len]
            
            tensor_text = torch.tensor([indices], dtype=torch.long).to(device)
            prediction = model(tensor_text)
            sentiment = "положительная" if prediction.item() > 0.5 else "отрицательная"
            print(f"Текст: '{text}' -> Тональность: {sentiment} ({prediction.item():.4f})")


if __name__ == "__main__":
    train_model()