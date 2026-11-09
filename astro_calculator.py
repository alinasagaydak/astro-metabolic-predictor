import numpy as np
import pandas as pd
from datetime import date, timedelta
from skyfield.api import load
from skyfield.framelib import ecliptic_frame


planets = load('de440s.bsp')

# Внутренние планеты и Земля
earth   = planets['earth']
sun     = planets['sun']
moon    = planets['moon']
mercury = planets['mercury']
venus   = planets['venus']

# Внешние планеты требуют ' barycenter'!
mars    = planets['mars barycenter']
jupiter = planets['jupiter barycenter']
saturn  = planets['saturn barycenter']
uranus  = planets['uranus barycenter']
neptune = planets['neptune barycenter']

ts = load.timescale()


##### ФАЗЫ ЛУНЫ #####
def moon_phase(d: date) -> float:
    """
    Вычисляет фазу Луны от 0 (новолуние) до 1 (полнолуние).

    Parameters
    ----------
    d : date
        Дата для расчёта.

    Returns
    -------
    float
        Фаза Луны в диапазоне [0, 1).
    """
    t = ts.utc(d.year, d.month, d.day)
    e = earth.at(t)

    # Эклиптическая долгота Луны и Солнца
    moon_lon = e.observe(moon).apparent().frame_latlon(ecliptic_frame)[1].degrees
    sun_lon = e.observe(sun).apparent().frame_latlon(ecliptic_frame)[1].degrees

    # Угловое расстояние (разность долгот)
    phase_angle = (moon_lon - sun_lon) % 360

    # Нормализуем в диапазон [0, 1)
    return phase_angle / 360.0

def moon_phase_cyclic(d: date) -> tuple[float, float]:
    """
    Возвращает циклическое кодирование фазы Луны через sin и cos.

    Returns
    -------
    tuple[float, float]
        (sin(2π·phase), cos(2π·phase))
    """
    phase = moon_phase(d)
    return np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)


##### РЕТРОГРАДНОСТЬ МЕРКУРИЯ #####
def mercury_retrograde(d: date) -> bool:
    """
    Определяет, является ли Меркурий ретроградным в заданную дату.

    Используется численное дифференцирование эклиптической долготы:
    если долгота уменьшается со временем — планета ретроградна.

    Parameters
    ----------
    d : date
        Дата для проверки.

    Returns
    -------
    bool
        True если Меркурий ретроградный, False иначе.
    """
    # Момент t0
    t0 = ts.utc(d.year, d.month, d.day)
    e0 = earth.at(t0)
    m0 = e0.observe(mercury).apparent().frame_latlon(ecliptic_frame)[1].degrees

    # Момент t1 (следующий день)
    d1 = d + timedelta(days=1)
    t1 = ts.utc(d1.year, d1.month, d1.day)
    e1 = earth.at(t1)
    m1 = e1.observe(mercury).apparent().frame_latlon(ecliptic_frame)[1].degrees

    # Разностная производная долготы (с корректной обработкой перехода через 360°)
    dlon = (m1 - m0 + 180) % 360 - 180

    # Если долгота уменьшается — планета ретроградна
    return dlon < 0


#####  УГЛОВЫЕ РАССТОЯНИЯ МЕЖДУ ПЛАНЕТАМИ #####
def planetary_angle(p1, p2, d: date) -> float:
    """
    Вычисляет угловое расстояние между двумя планетами.

    Parameters
    ----------
    p1, p2 : skyfield vector
        Векторы планет из эфемерид.
    d : date
        Дата для расчёта.

    Returns
    -------
    float
        Угловое расстояние в градусах [0, 180].
    """
    t = ts.utc(d.year, d.month, d.day)
    e = earth.at(t)

    lon1 = e.observe(p1).apparent().frame_latlon(ecliptic_frame)[1].degrees
    lon2 = e.observe(p2).apparent().frame_latlon(ecliptic_frame)[1].degrees

    # Кратчайшее угловое расстояние на окружности
    return abs((lon1 - lon2 + 180) % 360 - 180)

def mercury_venus_angle(d: date) -> float:
    """Угловое расстояние Меркурий-Венера."""
    return planetary_angle(mercury, venus, d)

def mars_saturn_angle(d: date) -> float:
    """Угловое расстояние Марс-Сатурн."""
    return planetary_angle(mars, saturn, d)


#####  ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ #####
def day_of_week(d: date) -> int:
    """День недели: 0=понедельник, 6=воскресенье."""
    return d.weekday()

def is_weekend(d: date) -> int:
    """Бинарный флаг: 1 если выходной, 0 если будний день."""
    return int(d.weekday() >= 5)


##### РАСЧЕТ ВСЕХ ПАРАМЕТРОВ #####
def compute_all_astro_features(d: date) -> dict:
    """
    Вычисляет все астрологические признаки для одной даты.

    Returns
    -------
    dict
        Словарь с астрологическими параметрами.
    """
    moon_sin, moon_cos = moon_phase_cyclic(d)

    return {
        'moon_phase': moon_phase(d),
        'moon_phase_sin': moon_sin,
        'moon_phase_cos': moon_cos,
        'mercury_retro': int(mercury_retrograde(d)),
        'angle_merc_ven': mercury_venus_angle(d),
        'angle_mars_sat': mars_saturn_angle(d),
        'day_of_week': day_of_week(d),
        'is_weekend': is_weekend(d),
    }

