import ast
import math
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dateutil.parser import parse
from loguru import logger
from typing import List


def is_time_format(input_string: str) -> bool:
    try:
        parse(input_string)
        return True
    except ValueError:
        return False


def is_list(input_string: str) -> bool:
    try:
        value = input_string.replace(';', ',')
        value = ast.literal_eval(value)
        return isinstance(value, list)
    except (ValueError, SyntaxError):
        return False


def is_time_column(column) -> bool:
    if all(column.apply(lambda x: is_time_format(str(x)) if pd.notnull(x) else False)):
        return True
    return False


def is_list_column(column) -> bool:
    if all(column.apply(lambda x: is_list(str(x)) if pd.notnull(x) else False)):
        return True
    return False


def load_and_describe_data(path_to_file: str):
    """
    функция возвращает словарь с типами столбцов, кол-вом строк и столбцов,
    а также статистику значений и отклонений. Пример:
    {
        "columns_types": {
            "column_name": int64 | string | float64 | datetime64[ns],
            ...
        },
        "num_columns": 11,
        "num_rows": 100,
        "stats": {
            "max_values": {
                "column_name": 10,
                ...
            },
            ...
        }
    }

    :param path_to_file: path to file
    :return: dict
    """
    try:
        df = pd.read_csv(path_to_file)
    except FileNotFoundError:
        logger.error(f'file with path "{path_to_file}" was not found')
        sys.exit(1)

    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

    num_rows, num_columns = df.shape
    print(f'Кол-во строк: {num_rows}\nКол-во столбцов: {num_columns}')
    result = {
        'num_rows': num_rows,
        'num_columns': num_columns,
        'columns_types': {},
        'stats': {},
    }

    print('Список всех столбцов и типов данных:')
    for column_name in df.columns:

        if df[column_name].dtype == 'object':
            if is_time_column(df[column_name].dropna()):
                df[column_name] = pd.to_datetime(df[column_name])
            elif is_list_column(df[column_name].dropna()):
                pass  # оставляем list как object
            else:
                df[column_name] = df[column_name].astype('string')

        column_type = df[column_name].dtype
        print(f'{column_name:20} | {column_type}')
        result['columns_types'][column_name] = column_type

    numeric_columns = df.select_dtypes(include='number').columns.tolist()

    min_values = df[numeric_columns].min()
    max_values = df[numeric_columns].max()
    mean_values = df[numeric_columns].mean()
    median_values = df[numeric_columns].median()
    std_values = df[numeric_columns].std()

    print("**Минимальные значения:**")
    print(min_values)
    print("**Максимальные значения:**")
    print(max_values)
    print("**Средние значения:**")
    print(mean_values)
    print("**Медианы:**")
    print(median_values)
    print("**Стандартные отклонения:**")
    print(std_values)

    result['stats']['min_values'] = min_values.to_dict()
    result['stats']['max_values'] = max_values.to_dict()
    result['stats']['mean_values'] = mean_values.to_dict()
    result['stats']['median_values'] = median_values.to_dict()
    result['stats']['std_values'] = std_values.to_dict()

    return result, df


def validate_numeric_columns(df: pd.core.frame.DataFrame, min_value: float, max_value: float) -> dict:
    """
    функция врзвращает словарь в формате
    {
        "тип ошибки": {
            "название столбца": {
                "index поля": int,
                "значение": int | float | nan
            }
        }
    }

    проверка идет на соотвествие типа, на отсутсвие Nan и вхождение значнеия в [min_value, max_value]

    :param df: dataframe
    :param min_value:
    :param max_value:
    :return:
    """
    errors = {
        'null_value': {},
        'incorrect_type': {},
        'incorrect_value_range': {}
    }
    numeric_columns = df.select_dtypes(include='number').columns

    for column_name in numeric_columns:
        errors['null_value'][column_name] = []
        errors['incorrect_type'][column_name] = []
        errors['incorrect_value_range'][column_name] = []

        for index, value in df[column_name].items():
            if isinstance(value, float):
                if math.isnan(value):
                    errors['null_value'][column_name].append({'index': index, 'value': value})
                    continue

            if not isinstance(value, (int, float)):
                errors['incorrect_type'][column_name].append({'index': index, 'value': value})
                continue

            if min_value > value or value > max_value:
                errors['incorrect_value_range'][column_name].append({'index': index, 'value': value})

    return errors


def compare_intensity_across_categories(df: pd.core.frame.DataFrame, types_to_show: List[str]):
    """

    :param df: dataframe
    :param types_to_show: список строк, состоящий из названий типов инцидентов (в csv поле types),
    которые надо оставить на графике (для большей наглядности)
    :return:
    """

    # Список колонок, которые нужно оставить
    columns_to_keep = ['type', 'events_count', 'end_time']
    # Удаление всех колонок, кроме указанных
    df = df[columns_to_keep]

    # Фильтрация данных по указанным типам
    df = df[df['type'].isin(types_to_show)]

    df = df.groupby([df['end_time'].dt.date, 'type'])['events_count'].sum().reset_index()

    plt.figure()
    sns.lineplot(data=df, x='end_time', y='events_count', hue='type', marker='o')

    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('Count of Types Over Time')
    plt.legend(title='Type')

    plt.show()


if __name__ == '__main__':
    path = '11_corrupted_incidents.csv'

    stats, parsed_dataframe = load_and_describe_data(path_to_file=path)
    validate_numeric_columns(
        df=parsed_dataframe,
        min_value=-100,
        max_value=100000
    )
    compare_intensity_across_categories(
        df=parsed_dataframe,
        types_to_show=['Удаление учетных данных', 'Публичный доступ', 'Фишинг'],
    )
