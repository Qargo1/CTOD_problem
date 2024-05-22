import csv
import math
import subprocess
import sys
import io

# Установить utf-8 как стандартную кодировку для stdout и stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])

# Установка необходимых библиотек
try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np

def read_data(filename):
    """
    Функция для чтения данных из CSV-файла.

    Args:
        filename (str): Путь к CSV-файлу.

    Returns:
        list: Список пар значений (сила, деформация).
    """
    data = []
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader)  # Пропустить заголовок
            for row in reader:
                try:
                    force, deformation = map(float, row)
                    data.append((force, deformation))
                except ValueError:
                    print(f"Ошибка обработки строки: {row}")
                    continue
    except FileNotFoundError:
        print("Файл не найден.")
        return []

    if not data:
        print("Ошибка: В CSV-файле нет данных!")

    return data


def calculate_angles(cathet1, cathet2):
    """
    Эта функция вычисляет острые углы прямоугольного треугольника по его катетам.

    Args:
    cathet1: Длина первого катета (float).
    cathet2: Длина второго катета (float).

    Returns:
    tuple: Два острых угла в градусах (A, B).
    
    Угол A при катете {cathet1}
    Угол B при катете {cathet2}
    """

    if cathet1 <= 0 or cathet2 <= 0:
        raise ValueError("Катеты должны быть больше 0.")

    # Вычисление острых углов с помощью тангенса:
    angle_a = math.degrees(math.atan(cathet2 / cathet1))
    angle_b = 90 - angle_a

    return angle_a, angle_b


def calculate_cathet(known_cathet, angle, opposite_angle):
    """
    Эта функция вычисляет один из катетов прямоугольного треугольника по значению другого катета, 
    двум углам и типу искомого катета.

    Args:
    known_cathet: Длина известного катета (float).
    angle: Значение угла, прилежащего к искомому катету (float, в градусах).
    opposite_angle: Значение угла, противолежащего искомому катету (float, в градусах).

    Returns:
    float: Длина искомого катета.
    """

    if known_cathet <= 0:
        raise ValueError("Известный катет должен быть больше 0.")

    if angle <= 0 or angle >= 90:
        raise ValueError("Значение угла должно быть в диапазоне от 0 до 90 градусов.")

    if opposite_angle <= 0 or opposite_angle >= 90:
        raise ValueError("Значение угла должно быть в диапазоне от 0 до 90 градусов.")

    if angle + opposite_angle != 90:
        raise ValueError("Сумма углов должна быть равна 90 градусам.")

    # Вычисление искомого катета с помощью синуса и косинуса:
    if angle == 90:
    # Если известен прилежащий катет и угол 90°, то искомый - противолежащий:
        return known_cathet / math.sin(math.radians(opposite_angle))
    else:
    # Если известен противолежащий катет и угол:
        return known_cathet * math.tan(math.radians(angle))


def calculate_vp(force_data, deformation_data):
    """
    Функция для подсчета vp графическим методом.

    Args:
        force_data (list): Список значений силы (Н).
        deformation_data (list): Список значений деформации (м).
        yield_stress (float): Напряжение при предельной текучести.
        yield_deformation (float): Деформация при предельной текучести.
        fracture_stress (float): Напряжение при разрушении.
        fracture_deformation (float): Деформация при разрушении.
        elastic_slope (float): Угол наклона линии упругой деформации.

    Returns:
        tuple: Кортеж из значений vp, yield_deformation, elastic_slope, yield_stress.
    """
    # Преобразуем списки в массивы NumPy
    force_data_np = np.array(force_data)
    deformation_data_np = np.array(deformation_data)
    force_deformation_data_np = np.array([force_data_np, deformation_data_np]).T

    # Initialize variables
    min_slope = np.inf  # Set initial minimum slope to positive infinity
    min_slope_idx = None

    # Iterate through all data points (except the last one)
    for i in range(len(force_data_np) - 1):
        # Calculate slope for current pair of points
        if deformation_data_np[i + 1] - deformation_data_np[i] == 0:
            continue  # Skip this iteration if the deformations are the same
            
        slope = (force_data_np[i + 1] - force_data_np[i]) / (deformation_data_np[i + 1] - deformation_data_np[i])

        # Check if current slope is smaller (in absolute value) than the minimum slope
        if abs(slope) < abs(min_slope):
            min_slope = slope
            min_slope_idx = i

    # Calculate the elastic limit stress and deformation based on the point with the minimum slope
    elastic_limit_stress = force_data_np[min_slope_idx + 1]
    elastic_limit_deformation = deformation_data_np[min_slope_idx + 1]

    # Print the results
    print("Elastic Limit Stress:", elastic_limit_stress)
    print("Elastic Limit Deformation:", elastic_limit_deformation)

    # Находим точку разрушения
    fracture_stress_index = len(force_deformation_data_np) - 1
    fracture_deformation = force_deformation_data_np[fracture_stress_index, 1]
    fracture_stress = force_deformation_data_np[fracture_stress_index, 0]

    # Определяем угол наклона линии упругой деформации
    # elastic_slope = (elastic_limit_stress - force_deformation_data_np[0, 0]) / (elastic_limit_deformation - force_deformation_data_np[0, 1])
    elastic_slope_stress, elastic_slope_deformation = calculate_angles(elastic_limit_stress, elastic_limit_deformation)
    
    # Определяем точку пересечения линии упругой деформации с горизонтальной линией,
    # проведенной из точки разрушения
    # vp_deformation = fracture_deformation - elastic_limit_stress / elastic_slope
    vp_deformation = calculate_cathet(fracture_stress, elastic_slope_stress, elastic_slope_deformation)
    
    # Print the results
    print("vp_deformation:", vp_deformation)

    # Рассчитываем vp
    vp = fracture_deformation - vp_deformation

    # Возвращаем кортеж с нужными значениями
    return deformation_data_np, force_data_np, fracture_stress, fracture_deformation, vp_deformation, vp


def plot_and_save_graph(deformation_data_np, force_data_np, fracture_stress, fracture_deformation, vp_deformation, vp, filename='graph.jpeg'):
    """
    Функция для построения графика зависимости силы от деформации и сохранения его в файл.

    Args:
        force_data (list): Список значений силы (Н).
        deformation_data (list): Список значений деформации (мм).
        vp (float): Значение vp (мм).
        yield_deformation (float): Деформация при предельной текучести.
        elastic_slope (float): Угол наклона линии упругой деформации.
        yield_stress (float): Напряжение при предельной текучести.
        filename (str): Путь к файлу для сохранения графика. Если None, график просто отображается.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(deformation_data_np, force_data_np, label='Экспериментальные данные')
    plt.xlabel('Деформация (мм)')
    plt.ylabel('Сила (Н)')

    # Горизонтальная линия из точки разрушения
    plt.axhline(y=fracture_stress, color='r', linestyle='--', label='Сила при разрушении')

    # Линия нуля
    plt.axhline(y=0, color='black', linestyle='--', label='Сила при разрушении')

    # Plot the extension line
    plt.plot([0, vp_deformation], [0, fracture_stress], color='g', linestyle='--', label='Упругая деформация')

    print("Parametr needed: ", vp)

    # Линия нахождения Vp
    plt.plot([vp, fracture_deformation], [0, fracture_stress], color='b', linestyle='--', label='Определение Vp')

    vp_label_x = vp  # Координата x точки описания (совпадает с x нижней точки)
    vp_label_y = -600  # Координата y точки описания (смещена вниз на 0.1)

    plt.text(
        vp_label_x,
        vp_label_y,
        f"Vp = {vp_label_x:.2f}",
        ha='center',
        va='bottom',
        color='b',
        fontsize=10,
    )
    plt.title("График зависимости силы от деформации")
    plt.legend()

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def main():
    # Считывание данных из CSV-файла
    data = read_data('data.csv')
    if not data:
        print("Не удалось загрузить данные из файла.")
        return
    
    force_data = [row[0] for row in data]
    deformation_data = [row[1] for row in data]

    # Вычисление vp и других значений
    deformation_data_np, force_data_np, fracture_stress, fracture_deformation, vp_deformation, vp = calculate_vp(force_data, deformation_data)
    
    print("fracture_stress:", fracture_stress)

    # Построение и сохранение графика
    plot_and_save_graph(deformation_data_np, force_data_np, fracture_stress, fracture_deformation, vp_deformation, vp, 'graph.jpeg')

    # Печать значения vp
    print(f"vp = {vp:.3f} мм")

if __name__ == "__main__":
    main()
