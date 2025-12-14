"""
Реализация модели нейрона Маккалока-Питтса (MCP)
"""

import numpy as np


class McCullochPittsNeuron:
    """
    Класс, реализующий нейрон Маккалока-Питтса
    
    Атрибуты:
        weights (list): Веса связей нейрона
        threshold (int): Порог активации
    """
    
    def __init__(self, weights, threshold):
        """
        Инициализация нейрона MCP
        
        Args:
            weights (list): Веса связей нейрона
            threshold (int): Порог активации
        """
        self.weights = np.array(weights)
        self.threshold = threshold
    
    def activate(self, inputs):
        """
        Функция активации нейрона MCP
        
        Args:
            inputs (list): Входные сигналы (0 или 1)
            
        Returns:
            int: Выходной сигнал (0 или 1)
        """
        # Преобразуем входы в numpy массив
        inputs = np.array(inputs)
        
        # Вычисляем взвешенную сумму входов
        weighted_sum = np.sum(inputs * self.weights)
        
        # Применяем пороговую функцию активации
        return 1 if weighted_sum >= self.threshold else 0


# Примеры использования нейрона MCP для реализации логических функций
def logical_and():
    """Реализация логической функции И"""
    print("Логическая функция И (AND)")
    
    # Для функции И с двумя входами:
    # Веса: [1, 1]
    # Порог: 2 (оба входа должны быть 1)
    and_neuron = McCullochPittsNeuron([1, 1], 2)
    
    # Тестовые случаи
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1)
    ]
    
    for inputs, expected in test_cases:
        result = and_neuron.activate(inputs)
        print(f"Входы: {inputs} -> Выход: {result} (Ожидаем: {expected})")
    print()


def logical_or():
    """Реализация логической функции ИЛИ"""
    print("Логическая функция ИЛИ (OR)")
    
    # Для функции ИЛИ с двумя входами:
    # Веса: [1, 1]
    # Порог: 1 (хотя бы один вход должен быть 1)
    or_neuron = McCullochPittsNeuron([1, 1], 1)
    
    # Тестовые случаи
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1)
    ]
    
    for inputs, expected in test_cases:
        result = or_neuron.activate(inputs)
        print(f"Входы: {inputs} -> Выход: {result} (Ожидаем: {expected})")
    print()


def logical_not():
    """Реализация логической функции НЕ"""
    print("Логическая функция НЕ (NOT)")
    
    # Для функции НЕ с одним входом:
    # Веса: [-1]
    # Порог: 0 (вход должен быть 0 для активации)
    not_neuron = McCullochPittsNeuron([-1], 0)
    
    # Тестовые случаи
    test_cases = [
        ([0], 1),
        ([1], 0)
    ]
    
    for inputs, expected in test_cases:
        result = not_neuron.activate(inputs)
        print(f"Входы: {inputs} -> Выход: {result} (Ожидаем: {expected})")
    print()


def main():
    """Основная функция для демонстрации работы нейрона MCP"""
    print("Модель нейрона Маккалока-Питтса")
    print("=" * 30)
    
    # Демонстрация логических функций
    logical_and()
    logical_or()
    logical_not()
    
    # Пример создания нейрона с пользовательскими параметрами
    print("Пользовательский нейрон:")
    custom_neuron = McCullochPittsNeuron([2, -1, 3], 2)
    result = custom_neuron.activate([1, 0, 1])
    print(f"Входы: [1, 0, 1], Веса: [2, -1, 3], Порог: 2 -> Выход: {result}")


if __name__ == "__main__":
    main()