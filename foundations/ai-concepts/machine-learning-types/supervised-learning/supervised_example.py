"""
Пример обучения с учителем (Supervised Learning)
Задача: Предсказание цены дома на основе его площади
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def supervised_learning_example():
    """Демонстрация обучения с учителем"""
    print("=== Обучение с учителем (Supervised Learning) ===")
    print("Пример: Предсказание цены дома на основе его площади")
    
    # Создаем синтетические данные
    np.random.seed(42)
    # Площадь домов (в кв. метрах)
    X = np.array([[50], [70], [90], [110], [130], [150], [170], [190], [210], [230]])
    # Цена домов (в тысячах долларов) с небольшим шумом
    y = np.array([150, 200, 250, 300, 350, 400, 450, 500, 550, 600]) + np.random.normal(0, 10, 10)
    
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Создаем модель линейной регрессии
    model = LinearRegression()
    
    # Обучаем модель
    model.fit(X_train, y_train)
    
    # Делаем предсказания на тестовой выборке
    y_pred = model.predict(X_test)
    
    # Оцениваем качество модели
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Коэффициент детерминации (R²): {r2:.2f}")
    
    # Делаем предсказание для нового дома
    new_house_area = np.array([[120]])
    predicted_price = model.predict(new_house_area)
    
    print(f"\nПлощадь нового дома: {new_house_area[0][0]} кв. м")
    print(f"Предсказанная цена: {predicted_price[0]:.2f} тыс. $")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='Обучающие данные')
    plt.scatter(X_test, y_test, color='red', label='Тестовые данные')
    plt.plot(X, model.predict(X), color='green', linewidth=2, label='Линия регрессии')
    plt.scatter(new_house_area, predicted_price, color='orange', s=100, label='Предсказание', marker='x')
    plt.xlabel('Площадь дома (кв. м)')
    plt.ylabel('Цена (тыс. $)')
    plt.title('Пример обучения с учителем: Линейная регрессия')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nХарактеристики обучения с учителем:")
    print("- Есть правильные ответы (метки)")
    print("- Алгоритм учится отображать входы в выходы")
    print("- Можно оценить качество модели")
    print("- Применяется для задач классификации и регрессии")

if __name__ == "__main__":
    supervised_learning_example()