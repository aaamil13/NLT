# fast_cosmo.py
import numpy as np
from numba import njit

C_KM_S = 299792.458  # km/s

@njit(fastmath=True, cache=True)
def hubble_function_numba(z, H0, Omega_m, Omega_r):
    """Компилирана Хъбъл функция."""
    E_sq = Omega_m * (1 + z)**3 + Omega_r * (1 + z)**4
    # Добавяме малка стойност, за да избегнем корен от 0
    return H0 * np.sqrt(E_sq + 1e-12)

@njit(fastmath=True, cache=True)
def angular_distance_integrand(z, H0, Omega_m, Omega_r):
    """Компилиран интегранд за изчисляване на разстояния."""
    return 1.0 / hubble_function_numba(z, H0, Omega_m, Omega_r)

# Numba не поддържа директно quad, затова ще симулираме с проста сума на Риман.
# За по-голяма точност, може да се използва библиотека като numba_scipy.
@njit(fastmath=True, cache=True)
def comoving_distance_numba(z_target, H0, Omega_m, Omega_r):
    """Компилирана функция за изчисляване на коподвижно разстояние (проста интеграция)."""
    n_steps = 1000  # Брой стъпки в интеграцията
    z_steps = np.linspace(0, z_target, n_steps)
    dz = z_steps[1] - z_steps[0]
    
    integral = 0.0
    for i in range(n_steps - 1):
        z_mid = (z_steps[i] + z_steps[i+1]) / 2.0
        integral += angular_distance_integrand(z_mid, H0, Omega_m, Omega_r)
    
    return C_KM_S * integral * dz

@njit(fastmath=True, cache=True)
def angular_diameter_distance_numba(z, H0, Omega_m, Omega_r):
    """Компилирана функция за ъглов диаметър."""
    return comoving_distance_numba(z, H0, Omega_m, Omega_r) / (1.0 + z)

@njit(fastmath=True, cache=True)
def sound_horizon_numba(H0, Omega_m, Omega_b):
    """Компилирана функция за звуков хоризонт (опростена формула за скорост)."""
    # Тази формула е често използвана апроксимация.
    # Заменете с точната, ако е нужно.
    omega_m_h2 = Omega_m * (H0 / 100.0)**2
    omega_b_h2 = Omega_b * (H0 / 100.0)**2
    return 44.5 * np.log(9.83 / omega_m_h2) / np.sqrt(1 + 10 * omega_b_h2**0.75) 