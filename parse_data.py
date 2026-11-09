import pandas as pd
from pathlib import Path

def parse_consumption_file(filepath: str, column_name: str) -> pd.DataFrame:
    """
    Парсит файл формата: ДД.ММ.ГГГГ\\tЧИСЛО_С_ЗАПЯТОЙ

    Parameters
    ----------
    filepath : str
        Путь к файлу.
    column_name : str
        Имя колонки для числовых данных ('coffee_cup' или 'beer_cup').

    Returns
    -------
    pd.DataFrame с колонками ['date', column_name]
    """
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Разделяем по табуляции
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Пропущена некорректная строка: {line!r}")
                continue

            date_str, value_str = parts
            # Заменяем запятую на точку (российский → питоновский формат)
            value_str = value_str.replace(',', '.')

            try:
                # Парсим дату в формате ДД.ММ.ГГГГ
                date = pd.to_datetime(date_str, format='%d.%m.%Y').date()
                value = float(value_str)
                records.append({'date': date, column_name: value})
            except ValueError as e:
                print(f"Ошибка парсинга строки {line!r}: {e}")

    return pd.DataFrame(records)


def build_dataset(coffee_path: str, beer_path: str) -> pd.DataFrame:
    """
    Собирает единый датасет из двух файлов.
    """
    df_coffee = parse_consumption_file(coffee_path, 'coffee_cup')
    df_beer = parse_consumption_file(beer_path, 'beer_cup')

    print(f"Записей о кофе: {len(df_coffee)}")
    print(f"Записей о пиве: {len(df_beer)}")

    # Outer join по дате
    df = pd.merge(df_coffee, df_beer, on='date', how='outer')

    # Заполняем пропуски нулями
    df['coffee_cup'] = df['coffee_cup'].fillna(0.0)
    df['beer_cup'] = df['beer_cup'].fillna(0.0)

    # Сортируем по дате (важно для временных рядов!)
    df = df.sort_values('date').reset_index(drop=True)

    return df

def main():
    coffee_path = 'coffee_data.txt'
    beer_path = 'beer_data.txt'
    output_path = 'consumption_dataset.csv'

    df = build_dataset(coffee_path, beer_path)
    df.to_csv(output_path, index=False)
    print(f"\nДатасет сохранён: {output_path}")

    print(f"\nИтоговый датасет: {len(df)} строк")
    print(f"Период: {df['date'].min()} — {df['date'].max()}")

    print(f"\nСтатистика по кофе:")
    print(f"   Всего записей с покупкой: {(df['coffee_cup'] > 0).sum()}")
    print(f"   Среднее количество чашек в день: {df[df['coffee_cup'] > 0]['coffee_cup'].mean():.2f}")
    print(f"   Максимальное количество чашек за день: {df['coffee_cup'].max():.2f}")

    print(f"\nСтатистика по пиву:")
    print(f"   Всего записей с покупкой: {(df['beer_cup'] > 0).sum()}")
    print(f"   Среднее количество кружек в день: {df[df['beer_cup'] > 0]['beer_cup'].mean():.2f}")
    print(f"   Максимальное количество кружек за день: {df['beer_cup'].max():.2f}")

if __name__ == '__main__':
    main()
