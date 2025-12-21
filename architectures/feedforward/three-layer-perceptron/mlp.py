"""
Реализация трехслойного персептрона (многослойного персептрона с одним скрытым слоем)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class ThreeLayerPerceptron:
    """
    Класс, реализующий трехслойный персептрон (MLP с одним скрытым слоем)
    
    Атрибуты:
        input_size (int): Количество входных нейронов
        hidden_size (int): Количество нейронов в скрытом слое
        output_size (int): Количество выходных нейронов
        learning_rate (float): Скорость обучения
        weights_input_hidden (numpy.ndarray): Веса между входным и скрытым слоем
        weights_hidden_output (numpy.ndarray): Веса между скрытым и выходным слоем
        bias_hidden (numpy.ndarray): Смещения скрытого слоя
        bias_output (numpy.ndarray): Смещения выходного слоя
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Инициализация трехслойного персептрона
        
        Args:
            input_size (int): Количество входных нейронов
            hidden_size (int): Количество нейронов в скрытом слое
            output_size (int): Количество выходных нейронов
            learning_rate (float): Скорость обучения (по умолчанию 0.1)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Инициализируем веса случайными значениями
        # Используем Xavier/Glorot инициализацию для лучшей сходимости
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        
        # Инициализируем смещения нулями
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)
    
    def sigmoid(self, x):
        """
        Сигмоидальная функция активации
        
        Args:
            x (numpy.ndarray): Входные значения
            
        Returns:
            numpy.ndarray: Значения после применения сигмоидальной функции
        """
        # Для предотвращения переполнения используем clip
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Производная сигмоидальной функции
        
        Args:
            x (numpy.ndarray): Входные значения
            
        Returns:
            numpy.ndarray: Значения производной сигмоидальной функции
        """
        return x * (1 - x)
    
    def forward(self, X):
        """
        Прямое распространение сигнала через сеть
        
        Args:
            X (numpy.ndarray): Входные данные
            
        Returns:
            tuple: (выход скрытого слоя, выход сети)
        """
        # Прямое распространение до скрытого слоя
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Прямое распространение до выходного слоя
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        
        return self.hidden_output, self.final_output
    
    def backward(self, X, y, hidden_output, final_output):
        """
        Обратное распространение ошибки
        
        Args:
            X (numpy.ndarray): Входные данные
            y (numpy.ndarray): Целевые значения
            hidden_output (numpy.ndarray): Выход скрытого слоя
            final_output (numpy.ndarray): Выход сети
        """
        # Вычисляем ошибку выходного слоя
        output_error = y - final_output
        output_delta = output_error * self.sigmoid_derivative(final_output)
        
        # Вычисляем ошибку скрытого слоя
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)
        
        # Обновляем веса и смещения
        self.weights_hidden_output += hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate
    
    def train(self, X, y, epochs=1000):
        """
        Обучение сети методом обратного распространения
        
        Args:
            X (numpy.ndarray): Входные данные
            y (numpy.ndarray): Целевые значения
            epochs (int): Количество эпох обучения
            
        Returns:
            list: История ошибок на каждой эпохе
        """
        errors = []
        
        for epoch in range(epochs):
            # Прямое распространение
            hidden_output, final_output = self.forward(X)
            
            # Вычисляем среднеквадратичную ошибку
            error = np.mean(np.square(y - final_output))
            errors.append(error)
            
            # Обратное распространение
            self.backward(X, y, hidden_output, final_output)
            
            # Выводим ошибку каждые 100 эпох
            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, Ошибка: {error}")
        
        return errors
    
    def predict(self, X):
        """
        Предсказание результата для новых данных
        
        Args:
            X (numpy.ndarray): Входные данные
            
        Returns:
            numpy.ndarray: Предсказанные значения
        """
        _, output = self.forward(X)
        return output


# Пример использования сети для решения задачи XOR
def xor_example():
    """Демонстрация работы сети на задаче XOR"""
    print("Решение задачи XOR с помощью трехслойного персептрона")
    print("=" * 50)
    
    # Создаем набор данных XOR
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    # Создаем сеть: 2 входа, 4 нейрона в скрытом слое, 1 выход
    mlp = ThreeLayerPerceptron(input_size=2, hidden_size=4, output_size=1, learning_rate=1.0)
    
    # Обучаем сеть
    print("Обучение сети...")
    errors = mlp.train(X, y, epochs=5000)
    
    # Тестируем сеть
    print("\nРезультаты:")
    for i in range(len(X)):
        prediction = mlp.predict(X[i].reshape(1, -1))
        print(f"Вход: {X[i]} -> Предсказание: {prediction[0][0]:.4f} (Цель: {y[i][0]})")
    
    # Построим график ошибки
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.title('Изменение ошибки во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Среднеквадратичная ошибка')
    plt.grid(True)
    plt.show()
    
    # Визуализируем границу принятия решений
    visualize_decision_boundary(mlp, X, y, "Граница принятия решений для задачи XOR")
    
    return mlp, errors


# Пример аппроксимации функции
def function_approximation_example():
    """Демонстрация аппроксимации функции синуса"""
    print("\nАппроксимация функции синуса")
    print("=" * 50)
    
    # Создаем данные для аппроксимации sin(x)
    X = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
    y = np.sin(X)
    
    # Нормализуем входные данные
    X_normalized = X / (2 * np.pi)
    
    # Создаем сеть: 1 вход, 10 нейронов в скрытом слое, 1 выход
    mlp = ThreeLayerPerceptron(input_size=1, hidden_size=10, output_size=1, learning_rate=0.5)
    
    # Обучаем сеть
    print("Обучение сети для аппроксимации sin(x)...")
    errors = mlp.train(X_normalized, y, epochs=2000)
    
    # Тестируем сеть
    test_X = np.linspace(0, 2*np.pi, 50).reshape(-1, 1)
    test_X_normalized = test_X / (2 * np.pi)
    predictions = mlp.predict(test_X_normalized)
    
    # Построим график
    plt.figure(figsize=(12, 6))
    
    # График функции и аппроксимации
    plt.subplot(1, 2, 1)
    plt.plot(X, y, 'b-', label='sin(x)', linewidth=2)
    plt.plot(test_X, predictions, 'ro-', label='Аппроксимация', markersize=4)
    plt.title('Аппроксимация функции sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # График ошибки
    plt.subplot(1, 2, 2)
    plt.plot(errors)
    plt.title('Изменение ошибки во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Среднеквадратичная ошибка')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return mlp, errors


def visualize_decision_boundary(mlp, X, y, title="Граница принятия решений"):
    """Визуализация границы принятия решений для классификационной задачи"""
    # Создаем сетку точек для визуализации
    h = 0.01  # шаг сетки
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Делаем предсказания для всех точек сетки
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Создаем цветовую карту
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    # Рисуем границу принятия решений
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    
    # Рисуем точки данных
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=cmap_bold, edgecolors='k', s=50)
    plt.colorbar(scatter)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('Вход 1')
    plt.ylabel('Вход 2')
    plt.grid(True)
    plt.show()


def circle_dataset_example():
    """Демонстрация работы сети на задаче классификации точек внутри и вне круга"""
    print("\nКлассификация точек внутри и вне круга")
    print("=" * 50)
    
    # Генерируем случайные точки
    np.random.seed(42)  # Для воспроизводимости результатов
    X = np.random.uniform(-1.5, 1.5, (300, 2))
    
    # Метки: 1 если точка внутри круга радиусом 1, 0 если вне
    distances = np.sqrt(np.sum(X**2, axis=1))
    y = (distances < 1).astype(int).reshape(-1, 1)
    
    # Разделяем на обучающую и тестовую выборки
    train_size = 250
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Создаем сеть: 2 входа, 10 нейронов в скрытом слое, 1 выход
    mlp = ThreeLayerPerceptron(input_size=2, hidden_size=10, output_size=1, learning_rate=0.5)
    
    # Обучаем сеть
    print("Обучение сети для классификации точек...")
    errors = mlp.train(X_train, y_train, epochs=3000)
    
    # Тестируем сеть
    predictions = mlp.predict(X_test)
    predicted_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_classes == y_test)
    print(f"\nТочность на тестовой выборке: {accuracy:.4f}")
    
    # Построим график ошибки
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.title('Изменение ошибки во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Среднеквадратичная ошибка')
    plt.grid(True)
    plt.show()
    
    # Визуализируем границу принятия решений
    visualize_decision_boundary(mlp, X_train, y_train, "Граница принятия решений для классификации круга")
    
    return mlp, errors


def main():
    """Основная функция для демонстрации работы трехслойного персептрона"""
    print("Трехслойный персептрон (MLP)")
    print("=" * 30)
    
    # Пример решения задачи XOR
    xor_example()
    
    # Пример аппроксимации функции
    function_approximation_example()
    
    # Пример классификации точек внутри и вне круга
    circle_dataset_example()


if __name__ == "__main__":
    main()