import subprocess
import sys
import io
import pandas as pd
import math
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Установить utf-8 как стандартную кодировку для stdout и stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])

def read_excel_files(directory):
    """
    Функция для чтения данных из CSV-файла (исправлено для XLSX).

    Args:
    filename (str): Путь к CSV-файлу (XLSX).

    Returns:
    list: Список пар значений (сила, деформация).
    """

    data_dict = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx'):
                file_path = os.path.join(root, file)
                try:
                    # Read the data
                    df = pd.read_excel(file_path)
                    data_dict[file] = df
                        
                except Exception as e:
                    print(f"Ошибка при загрузке файла {file}: {e}")

    return data_dict

def read_data(data_dict):

    data = []
    processed_data = {}

    for filename, data_frame in data_dict.items():
        for row in data_frame.itertuples(index=False):
            try:
                deformation, force = row[1], row[2]
                # Assuming 'Extension(mm)' is the second column, 'Force(N)' is the third

                try:
                    deformation = float(deformation)
                    force = abs(float(force))

                except:
                    print(f"Ошибка обработки строки: {row}")
                    continue

                if not isinstance(force, float) or not isinstance(deformation, float):
                    continue

                if force is None or deformation is None:
                    # Skip the row (adjust if needed)
                    continue

                if math.isnan(force) or math.isnan(deformation):
                    # Skip the row (adjust if needed)
                    continue
                
                if deformation <= 0:
                    # Skip the row (adjust if needed)
                    continue
                
                else:
                    data.append((force, deformation))

            except ValueError:
                print(f"Ошибка обработки строки: {row}")
                continue

        processed_data[filename] = data
        data = []
        
    return processed_data
        

def read_data_calculation(filename_calculation):
    """
    Функция для чтения данных из XLSX-файла.
    Для расчета CTOD
    """
    data_calculation = []
    try:
        # Read the data, skipping header rows (adjust if needed)
        df = pd.read_excel(filename_calculation)

        # Проверка наличия запятых (необязательно, если уверены в формате)
        if df.select_dtypes(include=['number']).apply(lambda x: x.astype(str).str.contains(',')).any().any():
            print("Запятые обнаружены в числовых данных.")

        # Замена запятых на точки в числовых столбцах
        df = df.apply(pd.to_numeric, errors='coerce')  
        # Преобразуем все столбцы в числовые, игнорируя нечисловые значения

        grip_length = df.iloc[2, 3]
        specimen_thickness = df.iloc[3, 3]
        specimen_width = df.iloc[4, 3]
        form_function = df.iloc[5, 3]
        poisson_ratio = df.iloc[6, 3]
        yield_strength = df.iloc[7, 3]
        elasticity = df.iloc[8, 3]
        initial_break_length = df.iloc[9, 3]
        length_from_force_to_break = df.iloc[11, 3]

    except FileNotFoundError:
        print("Файл не найден.")
        return []
    
    return grip_length, specimen_thickness, specimen_width, form_function, poisson_ratio, yield_strength, elasticity, initial_break_length, length_from_force_to_break

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def calculate_vp(force_data, deformation_data):
    force_data_np = np.array(force_data)
    deformation_data_np = np.array(deformation_data)

    df = pd.DataFrame({'force': force_data_np, 'deformation': deformation_data_np})
    window_size = 800
    mse_threshold = 1000  # Уменьшенный порог ошибки для более точного поиска
    
    best_start, best_end, best_mse = 0, window_size, float('inf')

    for i in range(len(df) - window_size):
        X_window = df['deformation'].iloc[i:i + window_size].values.reshape(-1, 1)
        y_window = df['force'].iloc[i:i + window_size].values
        
        model = LinearRegression()
        model.fit(X_window, y_window)
        
        predicted = model.predict(X_window)
        mse = mean_squared_error(y_window, predicted)
        
        if mse < best_mse and mse < mse_threshold:
            best_mse = mse
            best_start, best_end = i, i + window_size

    # Определяем линейный участок
    linear_region = df.iloc[best_start:best_end]
    X = linear_region['deformation'].values.reshape(-1, 1)
    y = linear_region['force'].values
    model = LinearRegression()
    model.fit(X, y)

    # Значения для vp
    max_force = max(force_data_np)
    fracture_stress = force_data_np[-1]
    fracture_deformation = deformation_data_np[-1]
    elastic_slope = model.coef_[0]
    vp_deformation = fracture_deformation - (fracture_stress / elastic_slope)

    if model.predict(X)[0] > (max_force * 0.15):
        return 'math_method_needed'

    return max_force, deformation_data_np, force_data_np, fracture_stress, fracture_deformation, vp_deformation, linear_region, model.predict(X)

def calculate_vp_variance(force_data, deformation_data, exclusion_ratio=0.75, linearity_threshold=0.95):
    force_data_np = np.array(force_data)
    deformation_data_np = np.array(deformation_data)
    
    # Строим датафрейм с данными для обучения модели
    df = pd.DataFrame({'force': force_data_np, 'deformation': deformation_data_np})

    window_size = 800
    variance_threshold = 1e-2  # порог дисперсии

    best_start, best_end, min_variance = 0, window_size, float('inf')

    # Перебираем окна, чтобы найти участок с минимальной дисперсией
    for i in range(len(df) - window_size):
        window_variance = df['force'].iloc[i:i + window_size].var()
        
        if window_variance < min_variance and window_variance < variance_threshold:
            min_variance = window_variance
            best_start, best_end = i, i + window_size

    # Первоначально найденный линейный участок
    linear_region = df.iloc[best_start:best_end]
    X = linear_region['deformation'].values.reshape(-1, 1)
    y = linear_region['force'].values

    # Обучение модели для расчета линейного участка
    model = LinearRegression()
    model.fit(X, y)

    # Проверка модели на линейность
    predicted_y = model.predict(X)
    r2 = r2_score(y, predicted_y)  # Коэффициент детерминации для оценки линейности

    # Если R^2 ниже порога, удаляем последние 40% данных и проверяем снова
    if r2 < linearity_threshold:
        cutoff_index = int(len(X) * (1 - exclusion_ratio))
        X, y = X[:cutoff_index], y[:cutoff_index]
        
        # Переобучение модели на укороченном наборе данных
        model.fit(X, y)
        predicted_y = model.predict(X)
        
        # Повторная проверка линейности
        r2 = r2_score(y, predicted_y)
        if r2 < linearity_threshold:
            print("Warning: Model does not meet linearity threshold even after truncation.")

    # Обновляем best_end после переобучения
    best_end = best_start + len(X)

    # vp значение
    max_force = max(force_data_np)
    fracture_stress = force_data_np[-1]
    fracture_deformation = deformation_data_np[-1]
    elastic_slope = model.coef_[0]
    vp_deformation = fracture_deformation - (fracture_stress / elastic_slope)
    vp = fracture_stress / elastic_slope

    return max_force, deformation_data_np, force_data_np, fracture_stress, fracture_deformation, vp_deformation, X, y, r2, best_end, vp

def calculate_ctod(grip_length, specimen_thickness, specimen_width, form_function, poisson_ratio, yield_strength, elasticity, initial_break_length, length_from_force_to_break, max_force, vp_deformation):

    a = ((max_force * grip_length) / (specimen_thickness * specimen_width ** 1.5) * form_function) ** 2
    b = ((1 - poisson_ratio) / (2 * yield_strength ** 2) / (2 * yield_strength * elasticity))
    c = (0.4 * (specimen_width - initial_break_length) * vp_deformation) / (0.4 * specimen_width + 0.6 * initial_break_length + length_from_force_to_break)

    return (a * b) + c

def plot_and_save_graph(ctod, max_force, deformation_data_np, force_data_np, vp_deformation, linear_region, predicted_force, filename):
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
    
    # Создаем сетку с двумя строками: график сверху, таблица снизу
    plt.figure(figsize=(10, 6))
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])

    # Создаем подграфик для графика
    ax = fig.add_subplot(gs[0])

    plt.plot(deformation_data_np, force_data_np, label='Экспериментальные данные')
    plt.xlabel('Удлинение (мм)')
    plt.ylabel('Сила (Н)')

    # Находим максимальное значение силы
    max_force_index = np.argmax(force_data_np)
    max_deformation_at_max_force = deformation_data_np[max_force_index]

    # Горизонтальная линия из точки разрушения
    plt.axhline(y=max_force, color='r', linestyle='--', label='Сила при разрушении')

    # Добавляем горизонтальную линию при нулевой силе
    plt.axhline(y=0, color='black', linestyle='-', label='Нулевая сила')

    # Линейный участок
    plt.plot(linear_region['deformation'], predicted_force, color='red', label='Линейный участок')

    # Построение параллельной линии для Vp
    elastic_slope = (predicted_force[-1] - predicted_force[0]) / (linear_region['deformation'].values[-1] - linear_region['deformation'].values[0])
    
    # Определение точки начала линии Vp
    vp_line_x = np.array([vp_deformation, max_deformation_at_max_force])
    vp_line_y = max_force - elastic_slope * (max_deformation_at_max_force - vp_line_x)

    plt.plot(vp_line_x, vp_line_y, 'b--', label='Определение Vp')

    plt.title("График зависимости силы от удлинения")
    plt.legend()

    vp_label_x = vp_deformation  # Координата x точки описания (совпадает с x нижней точки)

    # Устанавливаем пределы осей, чтобы график начинался с (0, 0)
    plt.xlim(left=min(0, np.min(deformation_data_np)))
    plt.ylim(bottom=min(0, np.min(force_data_np)))

    # Создаем данные для таблицы
    table_data = [
        ['Параметр', 'Значение'],
        ['CTOD', f'{ctod:.2f} мм'],
        ['Максимальная сила', f'{max_force:.2f} Н'],
        ['Удлиненение при max силе', f'{max_deformation_at_max_force:.2f} мм'],
        ['Vp', f'{vp_label_x:.2f} мм'],
        # Добавьте другие строки с необходимыми данными
    ]

    # Создаем таблицу
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')

    # Проверяем, создает ли таблицу
    try:
        table = ax_table.table(cellText=table_data,
                                cellLoc='center',  
                                colWidths=[0.5, 0.5],  
                                loc='center')  # Позиционирование таблицы по центру
        table.set_fontsize(12)
        table.auto_set_font_size(False)
        table.scale(1, 1.5)  # Увеличиваем высоту строк таблицы
    except Exception as e:
        print("Ошибка при создании таблицы:", e)

    if filename:
        # Создаем директорию, если она не существует
        save_dir = '../CTOD/graphs'
        os.makedirs(save_dir, exist_ok=True)

        # Формируем полный путь к файлу
        filepath = os.path.join(save_dir, filename)

        # Сохраняем график
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_and_save_graph_after_variance(ctod, max_force, deformation_data_np, force_data_np, fracture_stress, fracture_deformation, vp_deformation, X, y, r2, best_end, vp, filename):
    """
    Функция для построения графика зависимости силы от деформации и сохранения его в файл.

    Args:
        deformation_data_np (array): Массив значений деформации (мм).
        force_data_np (array): Массив значений силы (Н).
        fracture_stress (float): Значение силы при разрушении.
        fracture_deformation (float): Значение деформации при разрушении.
        vp_deformation (float): Значение Vp (мм).
        X (array): Окончательный массив значений деформации для линейного участка.
        y (array): Окончательный массив значений силы для линейного участка.
        r2 (float): Коэффициент детерминации модели на линейном участке.
        filename (str): Путь к файлу для сохранения графика.
    """

    # Создаем сетку с двумя строками: график сверху, таблица снизу
    plt.figure(figsize=(10, 6))
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])

    # Создаем подграфик для графика
    ax = fig.add_subplot(gs[0])

    # Находим максимальное значение силы
    max_force_index = np.argmax(force_data_np)
    max_deformation_at_max_force = deformation_data_np[max_force_index]

    # Координата x точки описания (совпадает с x нижней точки)
    vp_label_x = vp_deformation

    # Обрезаем экспериментальные данные, чтобы они начинались с конца линейного участка
    plt.plot(deformation_data_np[best_end:], force_data_np[best_end:], label='Экспериментальные данные', color='blue')

    plt.xlabel('Удлинение (мм)')
    plt.ylabel('Сила (Н)')

    # Линейный участок для упругой деформации
    plt.plot(X, y, color='green', linestyle='--', label=f'Линейный участок')

    # Горизонтальная линия из точки разрушения
    plt.axhline(y=fracture_stress, color='red', linestyle='--', label='Сила при разрушении')
    
    # Линия нахождения Vp
    plt.plot([vp_deformation, fracture_deformation], [0, fracture_stress], color='purple', linestyle='--', label='Определение Vp')

    plt.title("График зависимости силы от удлинения")
    plt.legend()

    # Устанавливаем пределы осей, чтобы график начинался с (0, 0)
    plt.xlim(left=min(0, np.min(deformation_data_np)))
    plt.ylim(bottom=min(0, np.min(force_data_np)))

    # Создаем данные для таблицы
    table_data = [
        ['Параметр', 'Значение'],
        ['CTOD', f'{ctod:.2f} мм'],
        ['Максимальная сила', f'{max_force:.2f} Н'],
        ['Удлинение при max силе', f'{max_deformation_at_max_force:.2f} мм'],
        ['Vp', f'{vp:.2f} мм'],
        # Добавьте другие строки с необходимыми данными
    ]

    # Создаем таблицу
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')

    # Проверяем, создает ли таблицу
    try:
        table = ax_table.table(cellText=table_data,
                                cellLoc='center',  
                                colWidths=[0.5, 0.5],  
                                loc='center')  # Позиционирование таблицы по центру
        table.set_fontsize(12)
        table.auto_set_font_size(False)
        table.scale(1, 1.5)  # Увеличиваем высоту строк таблицы
    except Exception as e:
        print("Ошибка при создании таблицы:", e)

    # Сохранение графика
    if filename:
        save_dir = '../CTOD/graphs'
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()
        plt.close()

def main():
    data_dict = read_excel_files('../CTOD/data')

    if not data_dict:
        print("Не удалось загрузить данные из файла данных испытания.")
        return
    
    processed_data = read_data(data_dict)

    for name, data in processed_data.items():

        grip_length, specimen_thickness, specimen_width, form_function, poisson_ratio, yield_strength, elasticity, initial_break_length, length_from_force_to_break = read_data_calculation('specimen/specimen_data.xlsx')

        force_data = [row[0] for row in data]
        deformation_data = [row[1] for row in data]

        file_name = f'graph_for_{name}.pdf'

        vp_result = calculate_vp(force_data, deformation_data)

        if vp_result != 'math_method_needed':
            # Вычисление vp и других значений
            max_force, deformation_data_np, force_data_np, fracture_stress, fracture_deformation, vp_deformation, linear_region, predicted_force = calculate_vp(force_data, deformation_data)
            
            # Извлечение значений для CTOD
            ctod = calculate_ctod(grip_length, specimen_thickness, specimen_width, form_function, poisson_ratio, yield_strength, elasticity, initial_break_length, length_from_force_to_break, max_force, vp_deformation)

            print("max_force:", max_force)

            # Построение и сохранение графика
            plot_and_save_graph(ctod, max_force, deformation_data_np, force_data_np, vp_deformation, linear_region, predicted_force, file_name)

            # Печать значения vp
            print(f"vp = {vp_deformation:.3f} мм")

            # Печать значения ctod
            print(f"ctod = {ctod:.3f} мм")

        else:
            # Считаем и рисуем график вторым способом
            file_name = f'graph_for_{name}_variance.pdf'

            max_force, deformation_data_np, force_data_np, fracture_stress, fracture_deformation, vp_deformation, X, y, r2, best_end, vp = calculate_vp_variance(force_data, deformation_data)
            
            # Извлечение значений для CTOD
            ctod = calculate_ctod(grip_length, specimen_thickness, specimen_width, form_function, poisson_ratio, yield_strength, elasticity, initial_break_length, length_from_force_to_break, max_force, vp)

            plot_and_save_graph_after_variance(ctod, max_force, deformation_data_np, force_data_np, fracture_stress, fracture_deformation, vp_deformation, X, y, r2, best_end, vp, file_name)

            print("max_force:", max_force)

            # Печать значения vp
            print(f"vp = {vp:.3f} мм")

if __name__ == "__main__":
    main()
