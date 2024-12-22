import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage import io, color
from skimage.transform import resize

# Загрузка и предобработка изображения
def load_image(filepath):
    # Загрузка изображения
    image = io.imread(filepath)  # Чтение изображения из файла
    # Преобразование в градации серого, если изображение цветное
    if len(image.shape) == 3:  # Проверка, является ли изображение цветным
        image = color.rgb2gray(image)
    # Изменение размера изображения для удобства обработки
    image = resize(image, (256, 256))
    return image

image_path = 'files1/test_image1.jpg'
original_image = load_image(image_path)

# Применение 2D вейвлет-преобразования
def apply_wavelet_transform(image, wavelet='db1'):
    # Выполняем 2D вейвлет-преобразование на 2 уровня
    coeffs = pywt.wavedec2(image, wavelet, level=2)
    # Преобразуем коэффициенты в массив для дальнейшей обработки
    coeffs_array, coeff_slices = pywt.coeffs_to_array(coeffs)
    return coeffs, coeffs_array, coeff_slices

coeffs, coeffs_array, coeff_slices = apply_wavelet_transform(original_image)

# Обнуление коэффициентов
def threshold_coefficients(coeffs_array, threshold):
    # Копируем массив коэффициентов
    coeffs_thresholded = coeffs_array.copy()
    # Зануляем коэффициенты, которые меньше заданного порога
    coeffs_thresholded[np.abs(coeffs_thresholded) < threshold] = 0
    return coeffs_thresholded

# Устанавливаем порог для сохранения 10% наиболее значимых коэффициентов
threshold_value = np.percentile(np.abs(coeffs_array), 90)
thresholded_coeffs_array = threshold_coefficients(coeffs_array, threshold_value)

# Выполнение обратного вейвлет-преобразования
def inverse_wavelet_transform(coeffs_array, coeff_slices, wavelet='db1'):
    # Восстанавливаем коэффициенты из массива
    coeffs_reconstructed = pywt.array_to_coeffs(coeffs_array, coeff_slices, output_format='wavedec2')
    # Применяем обратное вейвлет-преобразование для восстановления изображения
    reconstructed_image = pywt.waverec2(coeffs_reconstructed, wavelet)
    return reconstructed_image

reconstructed_image = inverse_wavelet_transform(thresholded_coeffs_array, coeff_slices)

# Анализ и визуализация результатов
def visualize_results(original, coeffs_array, thresholded_coeffs_array, reconstructed):
    # Отображение исходного изображения, коэффициентов и восстановленного изображения
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title("Оригинальное изображение")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Коэффициенты вейвлет-преобразования")
    plt.imshow(np.log1p(np.abs(coeffs_array)), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Обнуленные коэффициенты")
    plt.imshow(np.log1p(np.abs(thresholded_coeffs_array)), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Восстановленное изображение")
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Визуализация результатов
visualize_results(original_image, coeffs_array, thresholded_coeffs_array, reconstructed_image)

# Расчёт степени сжатия
compression_ratio = np.sum(thresholded_coeffs_array != 0) / np.sum(coeffs_array != 0)
print(f"Степень сжатия: {compression_ratio:.2f}")
