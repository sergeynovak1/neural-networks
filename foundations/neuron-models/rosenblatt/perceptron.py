"""
Реализация персептрона Розенблатта
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """
    Класс, реализующий персептрон Розенблатта
    
    Атрибуты:
        weights (numpy.ndarray): Веса связей нейрона
        bias (float): Порог активации (bias)
        learning_rate (float): Скорость обучения
        epochs (int): Максимальное количество эпох обучения
    """
    
    def __init__(self, input_size, learning_rate=1.0, epochs=100):
        """
        Инициализация персептрона
        
        Args:
            input_size (int): Количество входов
            learning_rate (float): Скорость обучения (по умолчанию 1.0)
            epochs (int): Максимальное количество эпох обучения (по умолчанию 100)
        """
        # Инициализируем веса малыми случайными значениями
        self.weights = np.random.uniform(-0.5, 0.5, input_size)
        # Инициализируем порог (bias) нулем
        self.bias = 0.0
        # Устанавливаем скорость обучения
        self.learning_rate = learning_rate
        # Устанавливаем максимальное количество эпох
        self.epochs = epochs
    
    def activation(self, x):
        """
        Пороговая функция активации
        
        Args:
            x (float): Взвешенная сумма входов
            
        Returns:
            int: Выходной сигнал (0 или 1)
        """
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        """
        Предсказание результата для заданных входов
        
        Args:
            inputs (list or numpy.ndarray): Входные сигналы
            
        Returns:
            int: Предсказанное значение (0 или 1)
        """
        # Преобразуем входы в numpy массив
        inputs = np.array(inputs)
        
        # Вычисляем взвешенную сумму входов минус порог
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        
        # Применяем функцию активации
        return self.activation(weighted_sum)
    
    def train(self, training_inputs, labels):
        """
        Обучение персептрона на обучающих данных
        
        Args:
            training_inputs (list of lists): Обучающие примеры
            labels (list): Правильные ответы для обучающих примеров
            
        Returns:
            list: История ошибок на каждой эпохе
        """
        # Преобразуем входы и метки в numpy массивы
        training_inputs = np.array(training_inputs)
        labels = np.array(labels)
        
        # Список для хранения истории ошибок
        errors_history = []
        
        # Цикл по эпохам обучения
        for epoch in range(self.epochs):
            # Счетчик ошибок на текущей эпохе
            errors = 0
            
            # Проходим по всем обучающим примерам
            for inputs, label in zip(training_inputs, labels):
                # Получаем предсказание персептрона
                prediction = self.predict(inputs)
                
                # Вычисляем ошибку
                error = label - prediction
                
                # Если была ошибка, обновляем веса и порог
                if error != 0:
                    errors += 1
                    # Обновляем веса
                    self.weights += self.learning_rate * error * inputs
                    # Обновляем порог (bias)
                    self.bias += self.learning_rate * error
            
            # Сохраняем количество ошибок на этой эпохе
            errors_history.append(errors)
            
            # Если на этой эпохе не было ошибок, завершаем обучение
            if errors == 0:
                print(f"Обучение завершено на эпохе {epoch + 1}")
                break
        
        return errors_history
    
    def get_weights(self):
        """
        Получение текущих весов персептрона
        
        Returns:
            tuple: Кортеж из весов и порога (weights, bias)
        """
        return self.weights.copy(), self.bias


# Примеры использования персептрона
def logical_and_training():
    """Обучение персептрона реализации логической функции И"""
    print("Обучение персептрона для функции И (AND)")
    
    # Создаем персептрон с 2 входами
    perceptron = Perceptron(2, learning_rate=0.1, epochs=10)
    
    # Обучающие данные для функции И
    training_inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    
    labels = [0, 0, 0, 1]  # Ожидаемые выходы для функции И
    
    # Обучаем персептрон
    errors_history = perceptron.train(training_inputs, labels)
    
    # Выводим результаты обучения
    print("Результаты обучения:")
    for inputs, label in zip(training_inputs, labels):
        prediction = perceptron.predict(inputs)
        print(f"Входы: {inputs} -> Предсказание: {prediction} (Ожидаем: {label})")
    
    # Выводим финальные веса
    weights, bias = perceptron.get_weights()
    print(f"Финальные веса: {weights}")
    print(f"Финальный порог (bias): {bias}")
    
    return errors_history


def logical_or_training():
    """Обучение персептрона реализации логической функции ИЛИ"""
    print("\nОбучение персептрона для функции ИЛИ (OR)")
    
    # Создаем персептрон с 2 входами
    perceptron = Perceptron(2, learning_rate=0.1, epochs=10)
    
    # Обучающие данные для функции ИЛИ
    training_inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    
    labels = [0, 1, 1, 1]  # Ожидаемые выходы для функции ИЛИ
    
    # Обучаем персептрон
    errors_history = perceptron.train(training_inputs, labels)
    
    # Выводим результаты обучения
    print("Результаты обучения:")
    for inputs, label in zip(training_inputs, labels):
        prediction = perceptron.predict(inputs)
        print(f"Входы: {inputs} -> Предсказание: {prediction} (Ожидаем: {label})")
    
    # Выводим финальные веса
    weights, bias = perceptron.get_weights()
    print(f"Финальные веса: {weights}")
    print(f"Финальный порог (bias): {bias}")
    
    return errors_history


def plot_errors_history(and_errors, or_errors):
    """Построение графика изменения ошибок во время обучения"""
    plt.figure(figsize=(10, 5))
    
    # График ошибок для функции И
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(and_errors) + 1), and_errors, marker='o')
    plt.title('Ошибки при обучении функции И')
    plt.xlabel('Эпоха')
    plt.ylabel('Количество ошибок')
    plt.grid(True)
    
    # График ошибок для функции ИЛИ
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(or_errors) + 1), or_errors, marker='o', color='orange')
    plt.title('Ошибки при обучении функции ИЛИ')
    plt.xlabel('Эпоха')
    plt.ylabel('Количество ошибок')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Основная функция для демонстрации работы персептрона"""
    print("Персептрон Розенблатта")
    print("=" * 30)
    
    # Обучение персептрона для функции И
    and_errors = logical_and_training()
    
    # Обучение персептрона для функции ИЛИ
    or_errors = logical_or_training()
    
    # Построение графиков изменения ошибок
    plot_errors_history(and_errors, or_errors)


if __name__ == "__main__":
    main()