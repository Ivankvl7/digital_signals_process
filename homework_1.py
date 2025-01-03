# 1. Реализуйте фильтр тиснения произвольного размера.
# 2. Обработка изображения матрицей свертки четного размера.
# 3. Какой вид должна иметь матрица свертки, чтобы не изменять изображение?
# Приведите пример.




# ===================================== TASK 1 =============================================================

# фильтр тиснения произвольного размера
# размеры ядра свертки принимаются с инпута через пробел

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imr


def proc(image, kernel, const=0):
    kernel_sum = kernel.sum()
    image = image / 255.
    const = const / 255.
    # получение размеров изображения и ядра для итерации по пикселам и весам
    i_height, i_width = image.shape[0], image.shape[1]
    k_width, k_height = kernel.shape[0], kernel.shape[1]

    # создание пустого изображения
    filtered = np.zeros_like(image)

    # Итерация по каждому (x, y) пикселу в изображении ...
    for y in range(i_height):
        for x in range(i_width):
            weighted_pixel_sum = 0

            # Итерация по каждому весу (kx, ky) в ядре, определенному выше ...
            # 'Центральный' вес в ядре итерпретируется как имеющий координаты (0, 0);
            # тогда координаты остальных весов в ядре будут такими:
            #
            #  [ (-1,-1),  (0,-1),  (1,-1)
            #    (-1, 0),  (0, 0),  (1, 0)
            #    (-1, 1),  (0, 1),  (1, 1) ]
            #
            # Таким образом, пиксель изображения с координатами[y,x] будет умножен на вес ядра[0,0]; анналогично,
            # пиксель[y-1,x] будет умножен на вес ядра[-1,0] и.т.д.
            # Значение отфильтрованного писеля это сумма этих произведений. Итак
            #
            #   weighted_pixel_sum = image[y-1,x-1] * kernel[-1,-1] +
            #                        image[y-1,x  ] * kernel[-1, 0] +
            #                        image[y-1,x+1] * kernel[-1, 1] +
            #                        image[y,  x-1] * kernel[ 0, 1] +
            #                        image[y,  x  ] * kernel[ 0, 0] +
            #                        etc.

            for ky in range(-k_height // 2 + 1, k_height // 2 + 1):
                for kx in range(-k_width // 2 + 1, k_width // 2 + 1):
                    pixel = 0
                    pixel_y = y - ky
                    pixel_x = x - kx

                    # проверка: если пиксель выходит за край изображения, то он равен нулю,
                    # а иначе он берется из изображения.
                    if (pixel_y >= 0) and (pixel_y < i_height) and (pixel_x >= 0) and (pixel_x < i_width):
                        pixel = image[pixel_y, pixel_x]

                    # текущая позия в ядре
                    pos = (ky + k_height // 2, kx + k_width // 2)
                    # import pdb; pdb.set_trace()
                    # получение веса ядра в текущей позиции
                    weight = kernel[pos[1], pos[0]]

                    weighted_pixel_sum += pixel * weight
            weighted_pixel_sum += const

            # наконец, пиксель с позицией (x,y) это сумма взвешенных соседних пикселей
            if weighted_pixel_sum > 1.:
                weighted_pixel_sum = 1.
            elif weighted_pixel_sum < 0.:
                weighted_pixel_sum = 0.
            # нормализация
            filtered[y, x] = weighted_pixel_sum

    return (filtered * 255).astype('int')


im = imr.imread('files1/test_image.jpg')  # сохраняем изображение в виде массива numpy
# print(im)
plt.rcParams["figure.figsize"] = (15, 10)
plt.rc('axes', facecolor=(1, 1, 1, 0), edgecolor=(1, 1, 1, 0))
plt.rc(('xtick', 'ytick'), color=(1, 1, 1, 0))
plt.rcParams["image.cmap"] = "gray"
# plt.imshow(im)
# plt.show()

# вводим высоту и ширину матрицы в одну строку через пробел
height, width = [int(num) for num in sys.stdin.readline().split()]
# height, width = 3, 3

# =============================================== TASK 2 =============================================================
# при вводе четной размерности (высоты или ширины) дополняем ее до нечетной
if height % 2 == 0:
    height += 1
if width % 2 == 0:
    width += 1
size = height * width

k = np.zeros((height, width))

# формируем ядро, дополненные строка и/или столбец остаются заполненными 0
for h in range(1, height):
    for w in range(1, width):
        if h < w:
            k[h][w] = 1
        elif h > w:
            k[h][w] = -1

# k = np.array([[0, 1,  0],
#              [-1,  0, 1],
#              [0,  -1,  0]])

print(k)
# ================================================ END TASK 2 ========================================================

im_out1 = proc(im, k, const=127)
plt.subplot(1, 3, 1)
plt.imshow(im_out1)
plt.show()

# ===================================================== END TASK 1 ===================================================

# ===================================================== TASK 3 =======================================================
# Чтобы матрица свертки не изменяла изображение, она должна быть единичной матрицей, например единичная матрица
# 3х3 выглядит так:
neutral_filter = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]])