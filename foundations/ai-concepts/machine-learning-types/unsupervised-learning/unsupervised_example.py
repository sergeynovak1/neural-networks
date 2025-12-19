"""
Пример обучения без учителя (Unsupervised Learning)
Задача: Группировка клиентов по покупательскому поведению
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def unsupervised_learning_example():
    """Демонстрация обучения без учителя"""
    print("=== Обучение без учителя (Unsupervised Learning) ===")
    print("Пример: Группировка клиентов по покупательскому поведению")
    
    # Создаем синтетические данные о клиентах
    # Ось X: количество покупок в месяц
    # Ось Y: средний чек
    np.random.seed(42)
    customers = np.vstack([
        np.random.normal(5, 1, (20, 2)) * [1, 10],   # Группа 1: мало покупок, низкий чек
        np.random.normal(15, 1, (20, 2)) * [1, 10],  # Группа 2: много покупок, высокий чек
        np.random.normal(10, 1, (20, 2)) * [1, 5]    # Группа 3: среднее количество, средний чек
    ])
    
    # Применяем алгоритм K-means для нахождения 3 кластеров
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(customers)
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green']
    for i in range(3):
        cluster_points = customers[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[i], label=f'Группа клиентов {i+1}', alpha=0.7, s=50)
    
    # Центры кластеров
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], 
               c='black', marker='x', s=200, linewidths=3, label='Центры кластеров')
    
    plt.xlabel('Количество покупок в месяц')
    plt.ylabel('Средний чек ($)')
    plt.title('Пример обучения без учителя: Кластеризация K-means')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Найденные группы клиентов:")
    for i in range(3):
        cluster_points = customers[clusters == i]
        avg_purchases = np.mean(cluster_points[:, 0])
        avg_amount = np.mean(cluster_points[:, 1])
        print(f"Группа {i+1}: {len(cluster_points)} клиентов, "
              f"в среднем {avg_purchases:.1f} покупок в месяц, "
              f"средний чек ${avg_amount:.2f}")
    
    print("\nХарактеристики обучения без учителя:")
    print("- Нет правильных ответов")
    print("- Алгоритм ищет скрытые структуры в данных")
    print("- Применяется для кластеризации, снижения размерности, обнаружения аномалий")
    print("- Качество трудно оценить объективно")

def unsupervised_dimensionality_reduction_example():
    """Пример снижения размерности"""
    print("\n=== Пример снижения размерности ===")
    print("Использование PCA для визуализации многомерных данных")
    
    # Создаем многомерные данные
    X, _ = make_blobs(n_samples=100, centers=3, n_features=4, random_state=42)
    
    # Для визуализации используем только первые два признака
    plt.figure(figsize=(12, 5))
    
    # Исходные данные (первые два признака)
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.7)
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title('Исходные данные (2D проекция)')
    plt.grid(True, alpha=0.3)
    
    # Применяем PCA для снижения размерности
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    # Данные после PCA
    plt.subplot(1, 2, 2)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='red', alpha=0.7)
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')
    plt.title('Данные после PCA (2 компоненты)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Сохранено информации после PCA: {pca.explained_variance_ratio_.sum():.2%}")

if __name__ == "__main__":
    unsupervised_learning_example()
    unsupervised_dimensionality_reduction_example()