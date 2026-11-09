"""
Сборка и обогащение датасета: от сырых чеков к астрофизическим признакам.
Кафедра прикладной алхимии и квантового похмелья, 2026.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Импортируем нашу функцию расчета эфемерид
# Убедитесь, что astro_calculator.py лежит в той же директории!
from astro_calculator import compute_all_astro_features

# ============================================================
# 1. ПАРСИНГ СЫРЫХ ДАННЫХ
# ============================================================

def parse_consumption_file(filepath: str, target_col: str) -> pd.DataFrame:
    """
    Читает файл формата 'ДД.ММ.ГГГГ\\tЧИСЛО_С_ЗАПЯТОЙ'.
    """
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                date_str, val_str = line.split('\t')
                # Парсим дату и приводим десятичную дробь к питоновскому формату
                date = pd.to_datetime(date_str, format='%d.%m.%Y').date()
                value = float(val_str.replace(',', '.'))
                records.append({'date': date, target_col: value})
            except Exception as e:
                print(f"⚠️ Ошибка парсинга строки '{line}': {e}")
                
    return pd.DataFrame(records)

# ============================================================
# 2. СБОРКА И ОБОГАЩЕНИЕ
# ============================================================

def build_full_dataset(coffee_path: str, beer_path: str, output_path: str):
    print("📥 Чтение сырых данных...")
    df_coffee = parse_consumption_file(coffee_path, 'coffee_cup')
    df_beer = parse_consumption_file(beer_path, 'beer_cup')
    
    print(f"   Найдено записей по кофе: {len(df_coffee)}")
    print(f"   Найдено записей по пиву: {len(df_beer)}")
    
    # Outer join: сохраняем все дни, даже если пили только кофе или только пиво
    df = pd.merge(df_coffee, df_beer, on='date', how='outer')
    
    # Заполняем пропуски нулями (день без покупок = 0 рублей)
    df['coffee_cup'] = df['coffee_cup'].fillna(0.0)
    df['beer_cup'] = df['beer_cup'].fillna(0.0)
    
    # Сортируем по времени (критично для временных рядов!)
    df = df.sort_values('date').reset_index(drop=True)
    
    # ============================================================
    # 3. РАСЧЕТ АСТРОЛОГИЧЕСКИХ ПАРАМЕТРОВ
    # ============================================================
    print("\n🌌 Расчет эфемерид JPL DE440s (это может занять минуту)...")
    
    # Список колонок, которые возвращает наша функция
    astro_cols = [
        'moon_phase',
        'moon_phase_sin',
        'moon_phase_cos',
        'mercury_retro',
        'angle_merc_ven',
        'angle_mars_sat',
        'day_of_week',
        'is_weekend',
    ]
    
    # Инициализируем пустые колонки
    for col in astro_cols:
        df[col] = np.nan
        
    # Проходимся по каждой дате и вызываем compute_all_astro_features
    total = len(df)
    for idx, row in df.iterrows():
        current_date = row['date']
        features = compute_all_astro_features(current_date)
        
        for col in astro_cols:
            df.at[idx, col] = features[col]
            
        # Прогресс-бар в консоль
        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"   Обработано {idx + 1}/{total} дат ({(idx + 1)/total*100:.0f}%)")
            
    # Приводим типы для строгости
    df['mercury_retro'] = df['mercury_retro'].astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)
    
    # Сохраняем результат
    df.to_csv(output_path, index=False)
    print(f"\n✅ Датасет успешно сохранен: {output_path}")
    
    # Краткая сводка для научного отчета
    print("\n📊 Статистика выборки:")
    print(f"   Период: {df['date'].min()} — {df['date'].max()}")
    print(f"   Дней с покупкой кофе: {(df['coffee_cup'] > 0).sum()}")
    print(f"   Дней с покупкой пива: {(df['beer_cup'] > 0).sum()}")
    print(f"   Дней с ретроградным Меркурием: {df['mercury_retro'].sum()}")
    
    # Подсветка аномалии
    max_beer_idx = df['beer_cup'].idxmax()
    print(f"\n⚠️ Обнаружена аномалия: {df.loc[max_beer_idx, 'beer_cup']} руб. на пиво "
          f"в дату {df.loc[max_beer_idx, 'date']} (возможно, коллективная гиперфиксация).")

# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == '__main__':
    build_full_dataset(
        coffee_path='coffee_data.txt',
        beer_path='beer_data.txt',
        output_path='consumption_dataset_full.csv'
    )
