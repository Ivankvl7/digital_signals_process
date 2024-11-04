# Самостоятельная работа№2. Сегментация изображений с помощью метода k-средних


import cv2
import numpy as np

# Загружаем изображение
image = cv2.imread('files1/test_image1.jpg')
# Конвертируем изображение в массив данных
data = image.reshape((-1, 3))
data = np.float32(data)

# Задаем критерий остановки и количество кластеров
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3  # Количество кластеров
attempts = 10  # Количество попыток для выбора наилучшего результата

# Применяем k-средние
ret, label, center = cv2.kmeans(data, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

# Преобразуем центроиды обратно в тип uint8
center = np.uint8(center)
# Создаем сегментированное изображение
segmented_image = center[label.flatten()]
segmented_image = segmented_image.reshape(image.shape)

# Сохраняем и показываем результат
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
