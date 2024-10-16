import numpy as np
import matplotlib.image as imr
import sys
import random
import colorsys
import matplotlib.pyplot as plt

# Функция, возвращающая случайный цвет
def random_color():
    h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
    r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
    return [r, g, b]


# Метрика (пункт 8)
def metric(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Рекурсивная функция добавления пикселя в сегмент
# Pc - приемлемый диапазон, x - координаты добавляемого писеля,
# seg - матрица сегментов, im - изображение,
# P - сумма яркостей пикселей в сегменте,
# n - количество пикселей в сегменте, n - номер сегмента
def append(Pc, x, seg, im, P, n, i):
    # Добаляемый пиксель помечается принадлежащим текущему сегменту
    seg[x[0], x[1]] = i
    n = n + 1  # Увеличивается количество пикселей в сегменте
    P = P + im[x[0], x[1], :]  # Яркость добавленого пикселя добавляется к сумме яркостей
    for w in range(-1, 2):  # Циклы, проходящие
        for h in range(-1, 2):  # восемь соседей добавленного пикселя
            if w == 0 and h == 0: continue  # но сам пиксел не является своим соседом
            cx = x[0] + w;
            cy = x[1] + h  # Координаты n-го соседа
            # и если они выходят за пределы изображения,
            # то переходим к следующему соседу
            if cx < 0 or cy < 0 or cx >= im.shape[0] or cy >= im.shape[1]: continue
            # Если n-й сосед уже входит в какой-либо сегмент,
            if seg[cx, cy] != 0: continue  # то он пропускается
            # Находим расстояние между базовой яркостью сегмента
            Pi = metric(P / n, im[cx, cy, :])  # и яркостью n-го соседа
            if Pi <= Pc:  # и если оно меньше или равно приемлемому
                try:
                    # добавляем n-ный сосед в текущий сегмент
                    seg, P, n = append(Pc, [cx, cy], seg, im, P, n, i)
                except Exception:
                    print("Достигнут лимит вызова рекурсий!")
    # возвращается матрица сегментов, сумма яркостей пикселей в сегменте
    return (seg, P, n)  # и количество пикселей в сегменте


# Функция динамической сегментации
# im - изображение, Pc - приемлемый диапазон (пункт 1)
def din_seg(im, Pc):
    # создается матрица сегментов
    seg = np.zeros([im.shape[0], im.shape[1]])
    si = 1  # счетчик сегментов
    while True:
        # Ищется первый пиксель, не принадлежащий
        x = np.where(seg == 0)  # какому-либо сегменту
        # Если такого нет
        if len(x[0]) == 0 or len(x[1]) == 0: break  # то выходим из цикла
        # Если есть, то добавляем его в текущий сегмент
        seg, _, _ = append(Pc, [x[0][0], x[1][0]], seg, im, [0., 0., 0.], 0, si)
        cl = random_color()  # Случайный цвет
        for i in range(im.shape[0]):  # которым мы раскрашиваем
            for j in range(im.shape[1]):  # каждый пиксель изображения
                # который принадлежит
                if seg[i, j] == si: im[i, j] = cl  # текущему сегменту
        si = si + 1
    # возвращаем количество сегментов
    return si


import sys
import random
import colorsys

im = imr.imread('files2/test_image1.jpg').copy()
im.setflags(write=True)
# Запрашивается большой стек
# resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
# и большой лимит вызова рекурсий
sys.setrecursionlimit(10 ** 6)
# Динамическая сегментация
P = 90.  # с приемлемым диапазоном равным
seg_count = din_seg(im, P)
print("Всего сегментов: " + str(seg_count))
plt.imshow(im)
plt.show()
