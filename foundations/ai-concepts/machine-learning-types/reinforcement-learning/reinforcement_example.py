"""
Пример обучения с подкреплением (Reinforcement Learning)
Задача: Агент учится перемещаться к цели
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class SimpleEnvironment:
    """
    Простая среда для демонстрации обучения с подкреплением
    Агент учится перемещаться по линии для получения награды
    """
    def __init__(self):
        self.position = 0  # Начальная позиция
        self.target = 5    # Целевая позиция
        
    def reset(self):
        """Сброс среды в начальное состояние"""
        self.position = 0
        return self.position
    
    def step(self, action):
        """
        Выполнить действие и получить результат
        
        Args:
            action (int): 0 - влево, 1 - вправо
            
        Returns:
            tuple: (новая позиция, награда, завершено ли)
        """
        # Выполняем действие
        if action == 0:  # Влево
            self.position = max(0, self.position - 1)
        else:  # Вправо
            self.position = min(10, self.position + 1)
        
        # Вычисляем награду
        if self.position == self.target:
            reward = 10  # Большой положительный бонус за достижение цели
            done = True
        elif self.position == 0 or self.position == 10:
            reward = -1  # Небольшой штраф за крайние позиции
            done = False
        else:
            # Награда зависит от расстояния до цели
            distance = abs(self.position - self.target)
            reward = -distance * 0.1  # Чем ближе к цели, тем меньше штраф
            done = False
            
        return self.position, reward, done

class SimpleAgent:
    """
    Простой агент для обучения с подкреплением
    Использует Q-learning для выбора оптимальных действий
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q-таблица: q_table[состояние][действие] = качество действия
        self.q_table = np.zeros((11, 2))  # 11 позиций, 2 действия
    
    def choose_action(self, state):
        """
        Выбрать действие на основе Q-таблицы
        
        Args:
            state (int): Текущее состояние (позиция)
            
        Returns:
            int: Выбранное действие (0 или 1)
        """
        # Эксплорация: случайное действие с вероятностью exploration_rate
        if random.random() < self.exploration_rate:
            return random.randint(0, 1)
        
        # Эксплуатация: лучшее известное действие
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Обновить Q-таблицу на основе полученного опыта
        
        Args:
            state (int): Предыдущее состояние
            action (int): Выполненное действие
            reward (float): Полученная награда
            next_state (int): Новое состояние
        """
        # Q-learning обновление
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

def reinforcement_learning_example():
    """
    Пример обучения с подкреплением
    """
    print("=== Обучение с подкреплением (Reinforcement Learning) ===")
    print("Пример: Агент учится перемещаться к цели")
    
    # Создаем среду и агента
    env = SimpleEnvironment()
    agent = SimpleAgent()
    
    # Параметры обучения
    episodes = 1000  # Количество эпизодов обучения
    rewards_history = []  # Для отслеживания прогресса
    
    # Обучение
    print("Обучение агента...")
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        # Один эпизод (пока не достигнем цели или не прервемся)
        while not done:
            # Агент выбирает действие
            action = agent.choose_action(state)
            
            # Выполняем действие в среде
            next_state, reward, done = env.step(action)
            
            # Агент учится на результате
            agent.update_q_table(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
    
    # Демонстрация обученного агента
    print("Демонстрация обученного агента:")
    state = env.reset()
    steps = 0
    path = [state]
    
    print(f"Начальная позиция: {state}")
    while state != env.target and steps < 15:  # Максимум 15 шагов
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        path.append(state)
        direction = "вправо" if action == 1 else "влево"
        print(f"Шаг {steps+1}: {direction} -> Позиция {state}")
        steps += 1
        
        if done:
            print("Цель достигнута!")
            break
    
    print(f"Путь агента: {' -> '.join(map(str, path))}")
    
    # Визуализация прогресса обучения
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.xlabel('Эпизод')
    plt.ylabel('Общая награда')
    plt.title('Прогресс обучения агента')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Показать оптимальную стратегию
    print("\nОптимальная стратегия (на основе Q-таблицы):")
    for position in range(11):
        left_q = agent.q_table[position][0]
        right_q = agent.q_table[position][1]
        best_action = "вправо" if right_q > left_q else "влево"
        print(f"Позиция {position}: {best_action} (Q-значения: влево={left_q:.2f}, вправо={right_q:.2f})")
    
    print("\nХарактеристики обучения с подкреплением:")
    print("- Агент учится взаимодействовать со средой")
    print("- Получает награды/штрафы за действия")
    print("- Цель - максимизировать долгосрочную награду")
    print("- Применяется в играх, робототехнике, управлении")

if __name__ == "__main__":
    reinforcement_learning_example()