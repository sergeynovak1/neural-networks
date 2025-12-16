import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class IzhikevichNeuron:
    """Модель нейрона Ижикевича"""
    def __init__(self, a=0.02, b=0.2, c=-65, d=8, v_peak=30):
        """
        Параметры модели:
        a, b, c, d - параметры модели
        v_peak - пиковое значение потенциала
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_peak = v_peak

        self.v = self.c  # Мембранный потенциал
        self.u = self.b * self.v  # Восстанавливающая переменная
        self.I = 0  # Входной ток

    def update(self, dt=1.0):
        """Обновление состояния нейрона"""
        # Дифференциальные уравнения модели Ижикевича
        dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + self.I
        du = self.a * (self.b * self.v - self.u)

        self.v += dv * dt
        self.u += du * dt

        # Проверка на спайк
        spiked = False
        if self.v >= self.v_peak:
            self.v = self.c
            self.u += self.d
            spiked = True

        return spiked, self.v

class Synapse:
    """Модель химического синапса"""
    def __init__(self, weight=5, delay=5, tau=10):
        """
        Параметры синапса:
        weight - сила связи
        delay - задержка передачи (в шагах)
        tau - временная константа спада
        """
        self.weight = weight
        self.delay = delay
        self.tau = tau

        # Буфер для задержанных спайков
        self.spike_buffer = []

    def process_spike(self, current_time):
        """Обработка спайка и возврат постсинаптического тока"""
        # Добавляем время спайка в буфер
        self.spike_buffer.append(current_time)

        # Рассчитываем постсинаптический ток
        current = 0
        for spike_time in self.spike_buffer:
            dt = current_time - spike_time - self.delay
            if dt >= 0:
                # Экспоненциальное затухание
                current += self.weight * np.exp(-dt / self.tau)

        # Удаляем старые спайки
        self.spike_buffer = [t for t in self.spike_buffer
                           if current_time - t < self.delay + 5*self.tau]

        return current

def simulate_network(num_neurons=3, simulation_time=1000, dt=1.0):
    """Симуляция сети нейронов Ижикевича с синаптическими связями"""

    # Создаем нейроны (разные типы для демонстрации)
    neurons = []
    neuron_types = [
        (0.02, 0.2, -65, 8),    # RS (регулярно спайкующий)
        (0.02, 0.25, -65, 2),   # IB (вспышка с последующим спайком)
        (0.1, 0.2, -65, 2)      # FS (быстро спайкующий)
    ]

    for i in range(num_neurons):
        params = neuron_types[i % len(neuron_types)]
        neurons.append(IzhikevichNeuron(*params))

    # Создаем синаптические связи (все-со-всеми)
    synapses = []
    for i in range(num_neurons):
        row = []
        for j in range(num_neurons):
            if i != j:  # Нет самовозбуждения
                # Сильные связи от нейрона 0 к другим для демонстрации синхронизации
                weight = 15 if i == 0 else 5
                row.append(Synapse(weight=weight, delay=10, tau=5))
            else:
                row.append(None)
        synapses.append(row)

    # Массивы для записи результатов
    time_points = np.arange(0, simulation_time, dt)
    voltages = np.zeros((num_neurons, len(time_points)))
    spikes = [[] for _ in range(num_neurons)]

    # Внешний входной ток (импульсы для запуска активности)
    def external_input(t, neuron_idx):
        if neuron_idx == 0:
            # Первый нейрон получает периодический вход
            return 20 * (1 + 0.5 * np.sin(2 * np.pi * t / 200))
        else:
            return 10

    # Основной цикл симуляции
    for t_idx, t in enumerate(time_points):
        for i in range(num_neurons):
            # Собираем входной ток от других нейронов
            synaptic_current = 0
            for j in range(num_neurons):
                if synapses[j][i] is not None and len(spikes[j]) > 0:
                    # Используем последний спайк
                    if spikes[j][-1] <= t:
                        synaptic_current += synapses[j][i].process_spike(t)

            # Устанавливаем общий входной ток
            neurons[i].I = external_input(t, i) + synaptic_current

            # Обновляем нейрон
            spiked, v = neurons[i].update(dt)

            # Записываем результаты
            voltages[i, t_idx] = v

            if spiked:
                spikes[i].append(t)

    return time_points, voltages, spikes

def plot_results(time_points, voltages, spikes):
    """Визуализация результатов симуляции"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    num_neurons = voltages.shape[0]

    # 1. Мембранные потенциалы
    ax1 = axes[0]
    colors = ['b', 'r', 'g', 'm', 'c']
    for i in range(num_neurons):
        ax1.plot(time_points, voltages[i],
                color=colors[i % len(colors)],
                alpha=0.7,
                label=f'Нейрон {i}')

    ax1.set_xlabel('Время (мс)')
    ax1.set_ylabel('Мембранный потенциал (мВ)')
    ax1.set_title('Мембранные потенциалы нейронов')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Растры спайков
    ax2 = axes[1]
    for i in range(num_neurons):
        if spikes[i]:
            ax2.eventplot(spikes[i],
                         lineoffsets=i+1,
                         colors=colors[i % len(colors)],
                         linewidths=2)

    ax2.set_xlabel('Время (мс)')
    ax2.set_ylabel('Нейрон')
    ax2.set_title('Спайковая активность (растр)')
    ax2.set_yticks(range(1, num_neurons + 1))
    ax2.set_yticklabels([f'Нейрон {i}' for i in range(num_neurons)])
    ax2.grid(True, alpha=0.3)

    # 3. Межспайковые интервалы для анализа синхронизации
    ax3 = axes[2]
    isi_means = []
    isi_stds = []

    for i in range(num_neurons):
        if len(spikes[i]) > 1:
            isi = np.diff(spikes[i])
            isi_means.append(np.mean(isi))
            isi_stds.append(np.std(isi))
        else:
            isi_means.append(0)
            isi_stds.append(0)

    x = range(num_neurons)
    ax3.bar(x, isi_means, yerr=isi_stds,
            capsize=5, alpha=0.7,
            color=colors[:num_neurons])
    ax3.set_xlabel('Нейрон')
    ax3.set_ylabel('Средний межспайковый интервал (мс)')
    ax3.set_title('Статистика спайков (меньший std = лучше синхронизация)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Нейрон {i}' for i in range(num_neurons)])
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Вывод коэффициентов вариации (мера синхронизации)
    print("\nКоэффициенты вариации межспайковых интервалов (CV):")
    print("(Меньше значение = более регулярная активность)")
    print("-" * 50)

    for i in range(num_neurons):
        if len(spikes[i]) > 1:
            isi = np.diff(spikes[i])
            cv = np.std(isi) / np.mean(isi)
            print(f"Нейрон {i}: CV = {cv:.3f} ({'синхронизирован' if cv < 0.5 else 'несинхронизирован'})")
        else:
            print(f"Нейрон {i}: недостаточно спайков для анализа")

def analyze_synchronization(spikes, window_size=50):
    """Анализ синхронизации с помощью кросс-корреляции"""
    if len(spikes) < 2:
        print("Нужно как минимум 2 нейрона для анализа синхронизации")
        return

    # Создаем бинарные последовательности спайков
    max_time = int(max([max(s) if s else 0 for s in spikes]))
    spike_trains = []

    for neuron_spikes in spikes:
        train = np.zeros(max_time + 1)
        for spike_time in neuron_spikes:
            if spike_time <= max_time:
                train[int(spike_time)] = 1
        spike_trains.append(train)

    # Вычисляем кросс-корреляцию между первыми двумя нейронами
    if len(spike_trains) >= 2:
        correlation = signal.correlate(spike_trains[0], spike_trains[1], mode='full')
        lags = signal.correlation_lags(len(spike_trains[0]), len(spike_trains[1]), mode='full')

        # Нормализуем корреляцию
        correlation = correlation / np.max(correlation)

        # Визуализируем кросс-корреляцию
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(lags, correlation, 'b-', linewidth=2)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Задержка (мс)')
        ax.set_ylabel('Корреляция')
        ax.set_title('Кросс-корреляция спайковых последовательностей (Нейрон 0 и Нейрон 1)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-100, 100)

        # Определяем максимальную корреляцию
        max_corr_idx = np.argmax(correlation)
        max_corr_lag = lags[max_corr_idx]
        max_corr_value = correlation[max_corr_idx]

        print(f"\nАнализ синхронизации:")
        print(f"Максимальная корреляция: {max_corr_value:.3f} при задержке {max_corr_lag} мс")

        if abs(max_corr_lag) < 20 and max_corr_value > 0.3:
            print("Нейроны демонстрируют значительную синхронизацию!")
        elif max_corr_value > 0.1:
            print("Обнаружена слабая синхронизация.")
        else:
            print("Синхронизация не обнаружена.")

        plt.show()

# Запуск симуляции
if __name__ == "__main__":
    print("Симуляция сети нейронов Ижикевича")
    print("=" * 50)

    # Параметры симуляции
    simulation_time = 1000  # мс
    num_neurons = 3

    print(f"Количество нейронов: {num_neurons}")
    print(f"Время симуляции: {simulation_time} мс")
    print("\nТипы нейронов:")
    print("1. RS (регулярно спайкующий) - синий")
    print("2. IB (вспышка с последующим спайком) - красный")
    print("3. FS (быстро спайкующий) - зеленый")

    # Запускаем симуляцию
    time_points, voltages, spikes = simulate_network(
        num_neurons=num_neurons,
        simulation_time=simulation_time,
        dt=1.0
    )

    # Визуализируем результаты
    plot_results(time_points, voltages, spikes)

    # Анализируем синхронизацию
    analyze_synchronization(spikes)

    print("\n" + "=" * 50)
    print("Замечания по синхронизации:")
    print("- Нейрон 0 получает периодический вход, что делает его пейсмейкером")
    print("- Сильные синаптические связи от нейрона 0 к другим способствуют синхронизации")
    print("- Сходные межспайковые интервалы и фазовая синхронизация видны на графиках")