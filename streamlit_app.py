"""
Космо-Бар: Интерактивный предсказатель потребления кофеина и этанола.
Кафедра прикладной алхимии и квантового похмелья, 2026.

Запуск: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ
# ============================================================

st.set_page_config(
    page_title="🌌 Космо-Бар Предсказатель",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .coffee-card {
        background: linear-gradient(135deg, #6f4e37 0%, #8b6f47 100%);
    }
    .beer-card {
        background: linear-gradient(135deg, #f1c40f 0%, #e67e22 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. ЗАГРУЗКА МОДЕЛИ И ДАННЫХ
# ============================================================

@st.cache_resource
def load_model():
    """Загружает обученную модель и список признаков."""
    model_path = 'best_cosmo_bar_pipeline.pkl'
    features_path = 'feature_columns.pkl'
    
    if not Path(model_path).exists():
        st.error(f"❌ Модель не найдена: {model_path}")
        st.info("Сначала запустите `multi_output_ml.py` для обучения модели.")
        return None, None
    
    if not Path(features_path).exists():
        st.error(f"❌ Список признаков не найден: {features_path}")
        return None, None
    
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    
    return model, feature_cols

@st.cache_resource
def load_astro_calculator():
    """Импортирует функции расчёта астрологических параметров."""
    try:
        from astro_calculator import compute_all_astro_features
        return compute_all_astro_features
    except ImportError:
        st.error("❌ Не удалось импортировать astro_calculator.py")
        return None

# Загрузка при старте
model, feature_cols = load_model()
compute_astro = load_astro_calculator()

# ============================================================
# 3. ЗАГОЛОВОК
# ============================================================

st.markdown('<h1 class="main-header">🌌 Космо-Бар Предсказатель</h1>', unsafe_allow_html=True)
st.markdown("### Предсказание потребления кофеина и этанола на основе астрономических эфемерид")
st.markdown("---")

# Проверка загрузки модели
if model is None or compute_astro is None:
    st.stop()

# ============================================================
# 4. БОКОВАЯ ПАНЕЛЬ
# ============================================================

st.sidebar.header("⚙️ Настройки")
st.sidebar.markdown("### Режим предсказания")

prediction_mode = st.sidebar.radio(
    "Выберите режим:",
    ["📅 Одна дата", "📆 Неделя вперёд", "📊 Диапазон дат"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 О модели")
st.sidebar.info(f"""
- **Алгоритм:** {type(model.named_steps['model']).__name__}
- **Признаков:** {len(feature_cols)}
- **Обучено на:** октябрь 2025 — октябрь 2026
""")

# ============================================================
# 5. ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# ============================================================

def predict_for_date(input_date: date) -> dict:
    """Делает предсказание для одной даты."""
    # Рассчитываем астрологические признаки
    astro_features = compute_astro(input_date)
    
    # Извлекаем нужные признаки в правильном порядке
    X = np.array([[astro_features[col] for col in feature_cols]])
    
    # Предсказание
    y_pred = model.predict(X)
    
    coffee_pred = max(0, float(y_pred[0, 0]))
    beer_pred = max(0, float(y_pred[0, 1]))
    
    return {
        'date': input_date,
        'coffee_cup': coffee_pred,
        'beer_cup': beer_pred,
        'astro': astro_features
    }

# ============================================================
# 6. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================

def get_recommendation(pred: dict) -> str:
    """Генерирует рекомендацию на основе предсказания."""
    coffee = pred['coffee_cup']
    beer = pred['beer_cup']
    astro = pred['astro']
    
    if coffee > 3.0 and beer > 5.0:
        return "⚠️ **Высокий риск гиперфиксации.** Рекомендуется Земфира в наушниках и горизонтальное положение."
    elif coffee > 3.0:
        return "☕ **День продуктивной апатии.** Запаситесь Nescafe Crema и включите матанализ."
    elif beer > 5.0:
        return "🍺 **Вечер обещает быть интересным.** Контролируйте метрику пространства-времени."
    elif astro['mercury_retro']:
        return "☿️ **Ретроградный Меркурий.** Избегайте важных решений и новых коктейлей."
    elif astro['is_weekend']:
        return "🎉 **Выходной день.** Метрика стабильна, можно расслабиться."
    else:
        return "✅ **Будний день.** Продолжайте анализ экспериментальных данных."

def get_moon_phase_text(moon_phase: float) -> str:
    """Возвращает текстовое описание фазы Луны."""
    if moon_phase < 0.125 or moon_phase >= 0.875:
        return "🌑 Новолуние"
    elif moon_phase < 0.375:
        return "🌓 Первая четверть"
    elif moon_phase < 0.625:
        return "🌕 Полнолуние"
    else:
        return "🌗 Последняя четверть"

# ============================================================
# 7. ОСНОВНОЙ ИНТЕРФЕЙС
# ============================================================

if prediction_mode == "📅 Одна дата":
    st.subheader("🔮 Предсказание на конкретную дату")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        input_date = st.date_input("Выберите дату:", value=date.today())
        predict_button = st.button("🔮 Рассчитать", type="primary", use_container_width=True)
    
    if predict_button:
        pred = predict_for_date(input_date)
        
        with col2:
            # Метрики
            st.markdown("### 📊 Прогноз потребления")
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.markdown(f"""
                <div class="metric-card coffee-card">
                    <h3>☕ Кофе</h3>
                    <h2>{pred['coffee_cup']:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"""
                <div class="metric-card beer-card">
                    <h3>🍺 Пиво</h3>
                    <h2>{pred['beer_cup']:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Астрологическая обстановка
            st.markdown("### 🌌 Астрологическая обстановка")
            astro = pred['astro']
            
            st.info(f"""
            - **Фаза Луны:** {get_moon_phase_text(astro['moon_phase'])} ({astro['moon_phase']:.2f})
            - **Меркурий:** {'☿️ Ретроградный' if astro['mercury_retro'] else '☿️ Директный'}
            - **День:** {'🎉 Выходной' if astro['is_weekend'] else '💼 Будний'}
            - **Угол Меркурий-Венера:** {astro['angle_merc_ven']:.1f}°
            """)
            
            # Рекомендация
            st.markdown("### 💡 Рекомендация")
            st.success(get_recommendation(pred))

elif prediction_mode == "📆 Неделя вперёд":
    st.subheader("📆 Прогноз на неделю")
    
    today = date.today()
    next_week = [today + timedelta(days=i) for i in range(7)]
    
    predictions = [predict_for_date(d) for d in next_week]
    df_week = pd.DataFrame(predictions)
    
    # Таблица
    st.markdown("### 📊 Таблица предсказаний")
    display_df = df_week[['date', 'coffee_rub', 'beer_rub']].copy()
    display_df['date'] = display_df['date'].apply(lambda x: x.strftime('%d.%m.%Y'))
    display_df.columns = ['Дата', '☕ Кофе (руб.)', '🍺 Пиво (руб.)']
    st.dataframe(display_df.round(2), use_container_width=True)
    
    # График
    st.markdown("### 📈 Визуализация")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_week['date'],
        y=df_week['coffee_rub'],
        mode='lines+markers',
        name='☕ Кофе',
        line=dict(color='#6f4e37', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_week['date'],
        y=df_week['beer_rub'],
        mode='lines+markers',
        name='🍺 Пиво',
        line=dict(color='#f1c40f', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='Прогноз потребления на неделю',
        xaxis_title='Дата',
        yaxis_title='Рубли',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Детальный отчёт
    st.markdown("### 📝 Детальный отчёт")
    for pred in predictions:
        with st.expander(f"{pred['date'].strftime('%d.%m.%Y')} ({get_moon_phase_text(pred['astro']['moon_phase'])})"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("☕ Кофе", f"{pred['coffee_rub']:.2f} руб.")
            with col2:
                st.metric("🍺 Пиво", f"{pred['beer_rub']:.2f} руб.")
            st.info(get_recommendation(pred))

elif prediction_mode == "📊 Диапазон дат":
    st.subheader("📊 Предсказание на диапазон дат")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input("Начальная дата:", value=date.today())
    with col2:
        end_date = st.date_input("Конечная дата:", value=date.today() + timedelta(days=14))
    with col3:
        st.write("")  # Отступ
        st.write("")  # Отступ
        calculate_button = st.button("🔮 Рассчитать", type="primary", use_container_width=True)
    
    if start_date > end_date:
        st.error("❌ Начальная дата должна быть раньше конечной!")
    elif calculate_button:
        date_range = pd.date_range(start=start_date, end=end_date).date
        predictions = [predict_for_date(d) for d in date_range]
        df_range = pd.DataFrame(predictions)
        
        # Таблица
        st.markdown("### 📊 Таблица предсказаний")
        display_df = df_range[['date', 'coffee_rub', 'beer_rub']].copy()
        display_df['date'] = display_df['date'].apply(lambda x: x.strftime('%d.%m.%Y'))
        display_df.columns = ['Дата', '☕ Кофе (руб.)', '🍺 Пиво (руб.)']
        st.dataframe(display_df.round(2), use_container_width=True)
        
        # Кнопка скачивания
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Скачать CSV",
            csv,
            f"predictions_{start_date}_{end_date}.csv",
            "text/csv",
            key='download-csv'
        )
        
        # График
        st.markdown("### 📈 Визуализация")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           subplot_titles=('☕ Кофе', '🍺 Пиво'))
        
        fig.add_trace(
            go.Scatter(x=df_range['date'], y=df_range['coffee_rub'], 
                      mode='lines+markers', name='Кофе',
                      line=dict(color='#6f4e37')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_range['date'], y=df_range['beer_rub'], 
                      mode='lines+markers', name='Пиво',
                      line=dict(color='#f1c40f')),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Статистика
        st.markdown("### 📊 Статистика периода")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.metric("Всего дней", len(df_range))
        with col_s2:
            st.metric("Средний кофе/день", f"{df_range['coffee_rub'].mean():.2f} руб.")
        with col_s3:
            st.metric("Среднее пиво/день", f"{df_range['beer_rub'].mean():.2f} руб.")

# ============================================================
# 8. ФУТЕР
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 Кафедра прикладной алхимии и квантового похмелья, 2026</p>
    <p>📊 Модель обучена на реальных данных потребления за октябрь 2025 — октябрь 2026</p>
    <p>🌌 Использованы эфемериды JPL DE440s</p>
</div>
""", unsafe_allow_html=True)
