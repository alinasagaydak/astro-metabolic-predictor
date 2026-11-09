"""
Обучение многомерной ML-модели для одновременного предсказания кофе и пива.
Кафедра прикладной алхимии и квантового похмелья, 2026.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================

# Очищенный список признаков (без мультиколлинеарных)
FEATURE_COLS = [
    'moon_phase_sin',
    'moon_phase_cos',
    'mercury_retro',
    'angle_merc_ven',
    'angle_mars_sat',
    'is_weekend',
]

TARGET_COLS = ['coffee_cup', 'beer_cup']

# ============================================================
# 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================

def load_and_prepare_data(path: str = 'consumption_dataset_full.csv') -> pd.DataFrame:
    """Загружает датасет и приводит типы."""
    df = pd.read_csv(path, parse_dates=['date'])
    df['date'] = df['date'].dt.date
    
    # Приведение типов
    for col in FEATURE_COLS + TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['mercury_retro'] = df['mercury_retro'].astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)
    
    print(f"📊 Загружено {len(df)} записей")
    print(f"   Период: {df['date'].min()} — {df['date'].max()}")
    print(f"   Записей с кофе: {(df['coffee_cup'] > 0).sum()}")
    print(f"   Записей с пивом: {(df['beer_cup'] > 0).sum()}")
    
    return df

# ============================================================
# 3. РАЗБИЕНИЕ ДАННЫХ (СЛУЧАЙНОЕ)
# ============================================================

def split_data_random(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Случайное разбиение на train/test.
    Стратификация по is_weekend для сохранения пропорций будней/выходных.
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=df['is_weekend']  # Сохраняем пропорции выходных
    )
    
    print(f"\n🎲 Случайное разбиение:")
    print(f"   Train: {len(train)} записей")
    print(f"   Test:  {len(test)} записей")
    
    return train, test

# ============================================================
# 4. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================================

def train_models(train: pd.DataFrame, test: pd.DataFrame):
    """Обучает несколько моделей и возвращает результаты."""
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COLS]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET_COLS]
    
    models = {
        'Ridge (Linear)': MultiOutputRegressor(Ridge(alpha=1.0)),
        'Random Forest': MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        ),
        'Gradient Boosting': MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        ),
    }
    
    results = {}
    
    for name, model in models.items():
        # Пайплайн: скейлинг + модель
        # Используем RobustScaler вместо StandardScaler для устойчивости к выбросам
        # (вспомните аномалию 14,96 руб. на пиво!)
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', model)
        ])
        
        # Обучение
        pipeline.fit(X_train, y_train)
        
        # Предсказание
        y_pred = pipeline.predict(X_test)
        
        # Метрики для каждого таргета
        metrics = {}
        for i, target in enumerate(TARGET_COLS):
            r2 = r2_score(y_test[target], y_pred[:, i])
            mae = mean_absolute_error(y_test[target], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[target], y_pred[:, i]))
            metrics[target] = {'r2': r2, 'mae': mae, 'rmse': rmse}
        
        # Feature importance (берем из первой модели в MultiOutputRegressor)
        base_model = pipeline.named_steps['model'].estimators_[0]
        importance = None
        if hasattr(base_model, 'feature_importances_'):
            importance = pd.Series(
                base_model.feature_importances_,
                index=FEATURE_COLS
            ).sort_values(ascending=False)
        elif hasattr(base_model, 'coef_'):
            # Для Ridge берем среднее абсолютное значение коэффициентов по всем таргетам
            coef_avg = np.mean([
                np.abs(est.coef_) 
                for est in pipeline.named_steps['model'].estimators_
            ], axis=0)
            importance = pd.Series(coef_avg, index=FEATURE_COLS).sort_values(ascending=False)
        
        results[name] = {
            'pipeline': pipeline,
            'y_pred': y_pred,
            'metrics': metrics,
            'importance': importance,
        }
        
        # Вывод метрик
        print(f"\n🔬 {name}:")
        for target, m in metrics.items():
            icon = '☕' if target == 'coffee_cup' else '🍺'
            print(f"   {icon} {target}: R² = {m['r2']:.3f}, MAE = {m['mae']:.2f}, RMSE = {m['rmse']:.2f}")
    
    return results

# ============================================================
# 5. СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
# ============================================================

def save_best_model(results: dict, output_dir: str = '.'):
    """Сохраняет лучшую модель по среднему R²."""
    # Выбираем лучшую модель по среднему R² по обоим таргетам
    best_name = max(
        results,
        key=lambda k: np.mean([
            results[k]['metrics'][t]['r2'] 
            for t in TARGET_COLS
        ])
    )
    
    best_pipeline = results[best_name]['pipeline']
    best_importance = results[best_name]['importance']
    
    # Сохраняем модель
    model_path = Path(output_dir) / 'best_cosmo_bar_pipeline.pkl'
    joblib.dump(best_pipeline, model_path)
    print(f"\n✅ Лучшая модель сохранена: {model_path}")
    
    # Сохраняем список признаков
    features_path = Path(output_dir) / 'feature_columns.pkl'
    joblib.dump(FEATURE_COLS, features_path)
    print(f"✅ Список признаков сохранен: {features_path}")
    
    # Выводим важность признаков
    if best_importance is not None:
        print(f"\n📊 Важность признаков ({best_name}):")
        for feat, imp in best_importance.items():
            print(f"   {feat}: {imp:.3f}")
    
    print(f"\n🏆 Лучшая модель: {best_name}")
    
    return best_name

# ============================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    print("="*70)
    print("ОБУЧЕНИЕ МНОГОМЕРНОЙ ML-МОДЕЛИ")
    print("Предсказание кофе и пива одновременно")
    print("="*70)
    
    # Загрузка данных
    df = load_and_prepare_data('consumption_dataset_full.csv')
    
    # Разбиение (случайное, не хронологическое!)
    train, test = split_data_random(df, test_size=0.2, random_state=42)
    
    # Обучение моделей
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    results = train_models(train, test)
    
    # Сохранение лучшей модели
    print("\n" + "="*70)
    print("СОХРАНЕНИЕ МОДЕЛИ")
    print("="*70)
    save_best_model(results, output_dir='.')
    
    print("\n" + "="*70)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*70)

if __name__ == '__main__':
    main()
