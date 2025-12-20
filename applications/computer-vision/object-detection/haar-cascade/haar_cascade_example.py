"""
Пример использования каскадного классификатора Хаара для обнаружения лиц

Этот пример демонстрирует:
1. Загрузку предобученного каскадного классификатора Хаара для обнаружения лиц
2. Обнаружение лиц на изображении
3. Отображение результатов обнаружения

Для работы примера требуются:
- OpenCV (cv2)
- NumPy
- Matplotlib (для отображения результатов)

Предобученные классификаторы можно скачать с официального репозитория OpenCV:
https://github.com/opencv/opencv/tree/master/data/haarcascades
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os

def download_haar_cascade():
    """
    Загрузка предобученного каскадного классификатора Хаара для обнаружения лиц
    """
    cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    cascade_path = "haarcascade_frontalface_default.xml"
    
    # Проверяем, существует ли файл классификатора
    if not os.path.exists(cascade_path):
        print("Загрузка каскадного классификатора Хаара...")
        try:
            urllib.request.urlretrieve(cascade_url, cascade_path)
            print("Классификатор успешно загружен!")
        except Exception as e:
            print(f"Ошибка при загрузке классификатора: {e}")
            print("Пожалуйста, скачайте файл вручную с:")
            print("https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")
            return None
    
    return cascade_path

def detect_faces_haar(image_path=None, use_webcam=False):
    """
    Обнаружение лиц с помощью каскадного классификатора Хаара
    
    Параметры:
    image_path (str): Путь к изображению для обработки
    use_webcam (bool): Использовать веб-камеру для обнаружения лиц в реальном времени
    """
    
    # Загрузка классификатора
    cascade_path = download_haar_cascade()
    if cascade_path is None:
        return
    
    # Создание объекта каскадного классификатора
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if not face_cascade.load(cascade_path):
        print("Ошибка загрузки классификатора!")
        return
    
    if use_webcam:
        # Использование веб-камеры
        cap = cv2.VideoCapture(0)
        
        print("Нажмите 'q' для выхода")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Преобразование в градации серого
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Обнаружение лиц
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,    # Коэффициент масштабирования
                minNeighbors=5,     # Минимальное количество соседей
                minSize=(30, 30),   # Минимальный размер лица
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Рисование прямоугольников вокруг лиц
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Отображение результата
            cv2.imshow('Обнаружение лиц - Haar Cascade', frame)
            
            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Освобождение ресурсов
        cap.release()
        cv2.destroyAllWindows()
        
    elif image_path and os.path.exists(image_path):
        # Обработка изображения из файла
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение лиц
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"Найдено лиц: {len(faces)}")
        
        # Рисование прямоугольников вокруг лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Отображение результата с помощью matplotlib
        # Преобразование BGR в RGB для правильного отображения в matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title(f'Обнаружение лиц - Haar Cascade (найдено {len(faces)} лиц)')
        plt.axis('off')
        plt.show()
        
    else:
        # Демонстрационный пример с синтетическим изображением
        print("Создание демонстрационного изображения...")
        
        # Создание синтетического изображения с "лицами"
        demo_img = np.zeros((300, 500, 3), dtype=np.uint8)
        demo_img[:] = (200, 200, 200)  # Серый фон
        
        # Рисование "лиц" - простых прямоугольников с базовыми признаками
        # Лицо 1
        cv2.rectangle(demo_img, (50, 50), (150, 150), (0, 0, 0), -1)  # Голова
        cv2.rectangle(demo_img, (70, 70), (90, 100), (255, 255, 255), -1)  # Глаза
        cv2.rectangle(demo_img, (110, 70), (130, 100), (255, 255, 255), -1)  # Глаза
        cv2.rectangle(demo_img, (80, 120), (120, 130), (255, 255, 255), -1)  # Рот
        
        # Лицо 2
        cv2.rectangle(demo_img, (200, 100), (300, 200), (0, 0, 0), -1)  # Голова
        cv2.rectangle(demo_img, (220, 120), (240, 150), (255, 255, 255), -1)  # Глаза
        cv2.rectangle(demo_img, (260, 120), (280, 150), (255, 255, 255), -1)  # Глаза
        cv2.rectangle(demo_img, (230, 170), (270, 180), (255, 255, 255), -1)  # Рот
        
        # Преобразование в градации серого
        gray = cv2.cvtColor(demo_img, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение лиц (на синтетическом изображении результат будет не идеальным)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,  # Уменьшено для демонстрации
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"Найдено лиц на демонстрационном изображении: {len(faces)}")
        
        # Рисование прямоугольников вокруг найденных лиц
        result_img = demo_img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Отображение результата
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.imshow(cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Исходное изображение')
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Результат обнаружения - Haar Cascade (найдено {len(faces)} лиц)')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

def explain_parameters():
    """
    Объяснение параметров метода detectMultiScale
    """
    print("""
    Параметры detectMultiScale:
    
    1. scaleFactor (по умолчанию 1.1):
       - Определяет, насколько уменьшается размер изображения при каждом масштабе
       - Значение 1.1 означает уменьшение на 10%
       - Меньшие значения дают более точные результаты, но замедляют работу
    
    2. minNeighbors (по умолчанию 3):
       - Определяет, сколько соседей должно быть у каждого прямоугольника для сохранения
       - Высокие значения приводят к меньшему количеству ложных срабатываний
       - Низкие значения могут привести к большему количеству ложных обнаружений
    
    3. minSize:
       - Минимальный возможный размер объекта
       - Все объекты меньше этого размера будут игнорироваться
       - Помогает ускорить обработку и уменьшить ложные срабатывания
    
    4. maxSize:
       - Максимальный возможный размер объекта
       - Все объекты больше этого размера будут игнорироваться
       - Редко используется, так как обычно хотим найти объекты любого размера
    
    5. flags:
       - Дополнительные флаги для настройки алгоритма
       - cv2.CASCADE_SCALE_IMAGE - рекомендуется для новых версий OpenCV
    """)

if __name__ == "__main__":
    print("=== Каскадный классификатор Хаара для обнаружения лиц ===")
    print()
    
    # Объяснение параметров
    explain_parameters()
    
    print("\nВыберите режим работы:")
    print("1. Демонстрационный пример")
    print("2. Обработка изображения из файла (укажите путь)")
    print("3. Обнаружение лиц с веб-камеры (нажмите 'q' для выхода)")
    
    choice = input("\nВведите номер режима (1-3): ").strip()
    
    if choice == "1":
        detect_faces_haar()
    elif choice == "2":
        image_path = input("Введите путь к изображению: ").strip()
        detect_faces_haar(image_path=image_path)
    elif choice == "3":
        detect_faces_haar(use_webcam=True)
    else:
        print("Неверный выбор. Запуск демонстрационного примера.")
        detect_faces_haar()