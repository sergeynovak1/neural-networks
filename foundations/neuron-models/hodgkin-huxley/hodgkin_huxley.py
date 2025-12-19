"""
Реализация модели нейрона Ходжкина-Хаксли
"""

import numpy as np
import matplotlib.pyplot as plt


class HodgkinHuxleyNeuron:
    """
    Класс, реализующий модель нейрона Ходжкина-Хаксли
    
    Атрибуты:
        Cm (float): Емкость мембраны (мкФ/см²)
        g_Na (float): Максимальная проводимость натрия (мСм/см²)
        g_K (float): Максимальная проводимость калия (мСм/см²)
        g_L (float): Проводимость утечки (мСм/см²)
        E_Na (float): Равновесный потенциал натрия (мВ)
        E_K (float): Равновесный потенциал калия (мВ)
        E_L (float): Равновесный потенциал утечки (мВ)
        t (float): Текущее время (мс)
        V (float): Мембранный потенциал (мВ)
        m (float): Переменная активации натриевых каналов
        h (float): Переменная инактивации натриевых каналов
        n (float): Переменная активации калиевых каналов
        dt (float): Шаг интегрирования (мс)
    """
    
    def __init__(self, dt=0.01):
        """
        Инициализация нейрона Ходжкина-Хаксли
        
        Args:
            dt (float): Шаг интегрирования в миллисекундах (по умолчанию 0.01)
        """
        # Биофизические параметры
        self.Cm = 1.0      # Емкость мембраны (мкФ/см²)
        self.g_Na = 120.0  # Максимальная проводимость натрия (мСм/см²)
        self.g_K = 36.0    # Максимальная проводимость калия (мСм/см²)
        self.g_L = 0.3     # Проводимость утечки (мСм/см²)
        self.E_Na = 50.0   # Равновесный потенциал натрия (мВ)
        self.E_K = -77.0   # Равновесный потенциал калия (мВ)
        self.E_L = -54.387 # Равновесный потенциал утечки (мВ)
        
        # Начальные условия
        self.t = 0.0       # Время (мс)
        self.V = -65.0     # Начальный мембранный потенциал (мВ)
        self.m = 0.05      # Начальное значение m
        self.h = 0.6       # Начальное значение h
        self.n = 0.325     # Начальное значение n
        
        self.dt = dt       # Шаг интегрирования
        self.I_ext = 0.0   # Внешний ток (мкА/см²)
    
    def alpha_m(self, V):
        """Скорость открытия натриевых каналов (активация)"""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    def beta_m(self, V):
        """Скорость закрытия натриевых каналов (активация)"""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def alpha_h(self, V):
        """Скорость открытия натриевых каналов (инактивация)"""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def beta_h(self, V):
        """Скорость закрытия натриевых каналов (инактивация)"""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def alpha_n(self, V):
        """Скорость открытия калиевых каналов"""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    def beta_n(self, V):
        """Скорость закрытия калиевых каналов"""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    def sodium_current(self, V, m, h):
        """Натриевый ток"""
        return self.g_Na * m**3 * h * (V - self.E_Na)
    
    def potassium_current(self, V, n):
        """Калиевый ток"""
        return self.g_K * n**4 * (V - self.E_K)
    
    def leak_current(self, V):
        """Ток утечки"""
        return self.g_L * (V - self.E_L)
    
    def update(self):
        """
        Обновление состояния нейрона на один шаг
        
        Returns:
            tuple: (время, мембранный потенциал)
        """
        # Вычисляем токи
        I_Na = self.sodium_current(self.V, self.m, self.h)
        I_K = self.potassium_current(self.V, self.n)
        I_L = self.leak_current(self.V)
        
        # Обновляем мембранный потенциал (основное уравнение Ходжкина-Хаксли)
        dVdt = (self.I_ext - I_Na - I_K - I_L) / self.Cm
        self.V += dVdt * self.dt
        
        # Обновляем переменные активации/инактивации
        dm_dt = self.alpha_m(self.V) * (1 - self.m) - self.beta_m(self.V) * self.m
        dh_dt = self.alpha_h(self.V) * (1 - self.h) - self.beta_h(self.V) * self.h
        dn_dt = self.alpha_n(self.V) * (1 - self.n) - self.beta_n(self.V) * self.n
        
        self.m += dm_dt * self.dt
        self.h += dh_dt * self.dt
        self.n += dn_dt * self.dt
        
        # Обновляем время
        self.t += self.dt
        
        return self.t, self.V
    
    def set_external_current(self, I_ext):
        """
        Установка внешнего тока
        
        Args:
            I_ext (float): Внешний ток в мкА/см²
        """
        self.I_ext = I_ext
    
    def simulate(self, duration, stimulus_start=0, stimulus_end=None, stimulus_amplitude=10):
        """
        Симуляция нейрона на заданное время
        
        Args:
            duration (float): Длительность симуляции в миллисекундах
            stimulus_start (float): Время начала стимула в миллисекундах (по умолчанию 0)
            stimulus_end (float): Время окончания стимула в миллисекундах (по умолчанию duration)
            stimulus_amplitude (float): Амплитуда стимула в мкА/см² (по умолчанию 10)
            
        Returns:
            tuple: (время, мембранный потенциал)
        """
        if stimulus_end is None:
            stimulus_end = duration
            
        steps = int(duration / self.dt)
        time_points = np.zeros(steps)
        voltage_points = np.zeros(steps)
        
        for i in range(steps):
            # Устанавливаем внешний ток в зависимости от времени
            if stimulus_start <= self.t <= stimulus_end:
                self.set_external_current(stimulus_amplitude)
            else:
                self.set_external_current(0.0)
            
            # Обновляем состояние нейрона
            t, V = self.update()
            time_points[i] = t
            voltage_points[i] = V
            
        return time_points, voltage_points


def plot_simulation(time_points, voltage_points):
    """
    Построение графика симуляции
    
    Args:
        time_points (numpy.ndarray): Временные точки
        voltage_points (numpy.ndarray): Значения мембранного потенциала
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, voltage_points, 'b-', linewidth=1)
    plt.xlabel('Время (мс)')
    plt.ylabel('Мембранный потенциал (мВ)')
    plt.title('Симуляция модели нейрона Ходжкина-Хаксли')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Пример использования
if __name__ == "__main__":
    # Создаем нейрон
    neuron = HodgkinHuxleyNeuron(dt=0.01)
    
    # Проводим симуляцию
    print("Запуск симуляции модели Ходжкина-Хаксли...")
    print("Параметры:")
    print("- Длительность: 50 мс")
    print("- Стимул: 10 мкА/см² с 5 по 30 мс")
    
    time_points, voltage_points = neuron.simulate(
        duration=50,
        stimulus_start=5,
        stimulus_end=30,
        stimulus_amplitude=10
    )
    
    # Выводим информацию о результатах
    print(f"\nСимуляция завершена.")
    print(f"Максимальный потенциал: {np.max(voltage_points):.2f} мВ")
    print(f"Минимальный потенциал: {np.min(voltage_points):.2f} мВ")
    
    # Строим график
    plot_simulation(time_points, voltage_points)