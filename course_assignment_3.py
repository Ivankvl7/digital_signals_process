# Самостоятельная работа№3. Сжатие изображений с использованием БПФ

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# сжатие изображения с помощью БПФ
def compress_image_fft(image_path, compression_ratio=0.5):
    # Загружаем картинку
    image = Image.open(image_path)
    image_data = np.array(image)

    # Если изображение цветное, применяем БПФ по каждому каналу отдельно
    if len(image_data.shape) == 3:
        channels = []
        for i in range(image_data.shape[2]):
            channel_data = image_data[:, :, i]

            # Применяем 2D БПФ
            fft_data = np.fft.fft2(channel_data)
            fft_shifted = np.fft.fftshift(fft_data)

            # Обрезаем коэффициенты для сжатия
            rows, cols = fft_shifted.shape
            row_mid, col_mid = rows // 2, cols // 2
            keep_rows, keep_cols = int(rows * compression_ratio), int(cols * compression_ratio)

            # обнуляем все высокие частоты
            compressed_fft = np.zeros_like(fft_shifted)
            compressed_fft[row_mid - keep_rows // 2: row_mid + keep_rows // 2,
            col_mid - keep_cols // 2: col_mid + keep_cols // 2] = fft_shifted[
                                                                  row_mid - keep_rows // 2: row_mid + keep_rows // 2,
                                                                  col_mid - keep_cols // 2: col_mid + keep_cols // 2]

            # Обратное преобразование Фурье
            compressed_fft_shifted = np.fft.ifftshift(compressed_fft)
            compressed_channel_data = np.fft.ifft2(compressed_fft_shifted).real
            channels.append(np.clip(compressed_channel_data, 0, 255).astype(np.uint8))

        # Объединяем каналы обратно в цветное изображение
        compressed_image_data = np.stack(channels, axis=2)
    else:
        # Применяем 2D БПФ для черно-белого изображения
        fft_data = np.fft.fft2(image_data)
        fft_shifted = np.fft.fftshift(fft_data)

        # Обрезаем коэффициенты для сжатия
        rows, cols = fft_shifted.shape
        row_mid, col_mid = rows // 2, cols // 2
        keep_rows, keep_cols = int(rows * compression_ratio), int(cols * compression_ratio)

        # обнуляем все высокие частоты
        compressed_fft = np.zeros_like(fft_shifted)
        compressed_fft[row_mid - keep_rows // 2: row_mid + keep_rows // 2,
        col_mid - keep_cols // 2: col_mid + keep_cols // 2] = fft_shifted[
                                                              row_mid - keep_rows // 2: row_mid + keep_rows // 2,
                                                              col_mid - keep_cols // 2: col_mid + keep_cols // 2]

        # Обратное преобразование Фурье
        compressed_fft_shifted = np.fft.ifftshift(compressed_fft)
        compressed_image_data = np.fft.ifft2(compressed_fft_shifted).real

    # Преобразуем обратно в картинку
    compressed_image = Image.fromarray(np.clip(compressed_image_data, 0, 255).astype(np.uint8))

    # Показываем оригинал и сжатую картинку
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Оригинал')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title('сжатое изображение')
    plt.imshow(compressed_image_data)
    plt.show()

    return compressed_image


# Пример использования функции
# коэффициент должен находиться в диапазоне от 0 до 1 (чем больше коэффициент тем меньше сжатие)
compressed_image = compress_image_fft('files1/test_image.jpg', compression_ratio=0.1)
compressed_image.save('files1/compressed_image.jpg')
