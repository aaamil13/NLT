#!/usr/bin/env python3
"""
Анизотропна нелинейна времева космология със забавяне на разширението по оси

Този модул реализира космологичен модел, където:
1. Разширението на Вселената е анизотропно (различно по различни посоки)
2. Времевото забавяне зависи от посоката/оста
3. Въвеждат се параметри за анизотропия

Математическа формулировка:
- Анизотропна метрика: ds² = -dt² + a₁²(t)dx² + a₂²(t)dy² + a₃²(t)dz²
- Посочно зависимо забавяне: τᵢ(z,θ,φ) = τ₀(z) × [1 + εᵢ×fᵢ(θ,φ)]
- Модифицирана Хъбъл функция: H(z,θ,φ) = H₀ × E(z) × G(z,θ,φ)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Настройка на логирането
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Физични константи
c = 299792458  # м/с - скорост на светлината


class AnisotropicNonlinearTimeCosmology:
    """
    Анизотропна нелинейна времева космология
    
    Този клас реализира космологичен модел с:
    - Анизотропно разширение по различни оси
    - Посочно зависимо времево забавяне
    - Модифицирани космологични разстояния
    """
    
    def __init__(self, 
                 # Стандартни космологични параметри
                 H0: float = 67.4, 
                 Omega_m: float = 0.315, 
                 Omega_Lambda: float = 0.685,
                 
                 # Нелинейно време параметри
                 alpha: float = 1.5, 
                 beta: float = 0.0, 
                 gamma: float = 0.5, 
                 delta: float = 0.1,
                 
                 # Анизотропни параметри
                 epsilon_x: float = 0.1,  # Анизотропия по x-ос
                 epsilon_y: float = 0.05, # Анизотропия по y-ос  
                 epsilon_z: float = 0.02, # Анизотропия по z-ос
                 
                 # Времево забавяне по оси
                 tau_x: float = 0.1,      # Забавяне по x-ос
                 tau_y: float = 0.05,     # Забавяне по y-ос
                 tau_z: float = 0.03,     # Забавяне по z-ос
                 
                 # Ъглови зависимости
                 phi_preference: float = 0.0,    # Предпочитана азимутна посока (радиани)
                 theta_preference: float = 0.0,  # Предпочитана полярна посока (радиани)
                 angular_strength: float = 1.0   # Сила на ъгловата зависимост
                 ):
        """
        Инициализация на анизотропната нелинейна времева космология
        
        Args:
            H0, Omega_m, Omega_Lambda: Стандартни космологични параметри
            alpha, beta, gamma, delta: Нелинейно време параметри
            epsilon_x,y,z: Анизотропни параметри по оси (0 = изотропно)
            tau_x,y,z: Времево забавяне по оси
            phi_preference, theta_preference: Предпочитани посоки
            angular_strength: Сила на анизотропията
        """
        
        # Стандартни параметри
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        
        # Нелинейно време
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Анизотропни параметри
        self.epsilon_x = epsilon_x
        self.epsilon_y = epsilon_y
        self.epsilon_z = epsilon_z
        
        # Времево забавяне
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.tau_z = tau_z
        
        # Ъглови зависимости
        self.phi_pref = phi_preference
        self.theta_pref = theta_preference
        self.angular_strength = angular_strength
        
        logger.info(f"Инициализирана анизотропна нелинейна времева космология:")
        logger.info(f"  Стандартни: H₀={H0:.1f}, Ωₘ={Omega_m:.3f}, ΩΛ={Omega_Lambda:.3f}")
        logger.info(f"  Нелинейно време: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}")
        logger.info(f"  Анизотропия: εₓ={epsilon_x:.3f}, εᵧ={epsilon_y:.3f}, εᵧ={epsilon_z:.3f}")
        logger.info(f"  Забавяне: τₓ={tau_x:.3f}, τᵧ={tau_y:.3f}, τᵧ={tau_z:.3f}")
        
    def anisotropic_factor(self, theta: float, phi: float) -> Tuple[float, float, float]:
        """
        Изчисляване на анизотропните фактори по оси
        
        Args:
            theta: Полярен ъгъл (0 до π)
            phi: Азимутен ъгъл (0 до 2π)
            
        Returns:
            Tuple с анизотропни фактори (f_x, f_y, f_z)
        """
        
        # Проекции на единичния вектор по координатните оси
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        # Компоненти на единичния вектор
        n_x = sin_theta * cos_phi
        n_y = sin_theta * sin_phi  
        n_z = cos_theta
        
        # Предпочитани посоки
        sin_theta_pref = np.sin(self.theta_pref)
        cos_theta_pref = np.cos(self.theta_pref)
        sin_phi_pref = np.sin(self.phi_pref)
        cos_phi_pref = np.cos(self.phi_pref)
        
        # Единичен вектор на предпочитаната посока
        n_pref_x = sin_theta_pref * cos_phi_pref
        n_pref_y = sin_theta_pref * sin_phi_pref
        n_pref_z = cos_theta_pref
        
        # Скаларно произведение с предпочитаната посока
        dot_product = n_x * n_pref_x + n_y * n_pref_y + n_z * n_pref_z
        
        # Анизотропни фактори
        f_x = 1.0 + self.epsilon_x * (n_x**2) * (1 + self.angular_strength * dot_product)
        f_y = 1.0 + self.epsilon_y * (n_y**2) * (1 + self.angular_strength * dot_product)
        f_z = 1.0 + self.epsilon_z * (n_z**2) * (1 + self.angular_strength * dot_product)
        
        return f_x, f_y, f_z
    
    def directional_time_delay(self, z: np.ndarray, theta: float, phi: float) -> np.ndarray:
        """
        Посочно зависимо времево забавяне
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            Модифицирана времева функция τ(z,θ,φ)
        """
        z = np.asarray(z)
        
        # Базова нелинейна времева функция
        one_plus_z = 1 + z
        z_safe = np.maximum(z, 1e-10)
        
        # Базова времева функция
        t_base = (self.alpha * z_safe**self.beta * 
                 np.exp(-self.gamma * z_safe) / one_plus_z + 
                 self.delta * np.log(one_plus_z))
        
        # Анизотропни фактори
        f_x, f_y, f_z = self.anisotropic_factor(theta, phi)
        
        # Проекции на посоката
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        # Единичен вектор на наблюдението
        n_x = sin_theta * cos_phi
        n_y = sin_theta * sin_phi
        n_z = cos_theta
        
        # Посочно зависимо забавяне
        delay_x = self.tau_x * (n_x**2) * f_x * np.exp(-z_safe/10)
        delay_y = self.tau_y * (n_y**2) * f_y * np.exp(-z_safe/10)  
        delay_z = self.tau_z * (n_z**2) * f_z * np.exp(-z_safe/10)
        
        total_delay = delay_x + delay_y + delay_z
        
        # Модифицирана времева функция
        t_modified = t_base * (1 + total_delay)
        
        # Проверка за NaN/inf
        if np.any(~np.isfinite(t_modified)):
            logger.warning("NaN/inf в анизотропната времева функция!")
            t_modified = np.where(np.isfinite(t_modified), t_modified, t_base)
            
        return t_modified
    
    def anisotropic_hubble_function(self, z: np.ndarray, theta: float, phi: float) -> np.ndarray:
        """
        Анизотропна Хъбъл функция H(z,θ,φ)
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            H(z,θ,φ) в km/s/Mpc
        """
        z = np.asarray(z)
        
        # Стандартна E(z) функция
        E_z_standard = np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_Lambda)
        
        # Анизотропна времева корекция
        t_z = self.directional_time_delay(z, theta, phi)
        
        # Анизотропни фактори
        f_x, f_y, f_z = self.anisotropic_factor(theta, phi)
        
        # Усреднен анизотропен фактор
        f_avg = (f_x + f_y + f_z) / 3.0
        
        # Анизотропна корекция
        anisotropic_correction = f_avg * (1 + self.alpha * t_z)
        
        # Модифицирана Хъбъл функция
        H_z = self.H0 * E_z_standard * anisotropic_correction
        
        return H_z
    
    def anisotropic_angular_diameter_distance(self, z: np.ndarray, theta: float, phi: float) -> np.ndarray:
        """
        Анизотропно ъглово диаметрово разстояние
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            D_A(z,θ,φ) в Mpc
        """
        z = np.asarray(z)
        
        def integrand(z_val):
            H_z = self.anisotropic_hubble_function(z_val, theta, phi)
            return c / (H_z * 1000)  # Конвертиране в Mpc
        
        D_A = np.zeros_like(z)
        
        for i, z_val in enumerate(z.flat):
            if z_val > 0:
                try:
                    integral, _ = integrate.quad(integrand, 0, z_val)
                    D_A.flat[i] = integral / (1 + z_val)
                except:
                    # Fallback към стандартно разстояние
                    D_A.flat[i] = c * z_val / (self.H0 * 1000 * (1 + z_val))
            else:
                D_A.flat[i] = 0
                
        return D_A.reshape(z.shape)
    
    def anisotropic_sound_horizon(self, z_star: float = 1100, theta: float = 0, phi: float = 0) -> float:
        """
        Анизотропен звуков хоризонт
        
        Args:
            z_star: Червено отместване на рекомбинацията
            theta: Полярен ъгъл на наблюдение
            phi: Азимутен ъгъл на наблюдение
            
        Returns:
            r_s(z*,θ,φ) в Mpc
        """
        
        def integrand(z):
            # Скорост на звука в барион-фотонна плазма
            Omega_b = 0.049
            Omega_gamma = 8.24e-5
            R_ratio = (3 * Omega_b) / (4 * Omega_gamma * (1 + z))
            c_s = c * np.sqrt(1 / (3 * (1 + R_ratio)))
            
            # Анизотропна Хъбъл функция
            H_z = self.anisotropic_hubble_function(z, theta, phi)
            
            return c_s / (H_z * 1000)  # Конвертиране в Mpc
        
        try:
            r_s, error = integrate.quad(integrand, z_star, 3000,
                                      epsabs=1e-10, epsrel=1e-8)
            
            if error > 0.01 * abs(r_s):
                logger.warning(f"Висока грешка в анизотропния звуков хоризонт: {error:.2e}")
                
            logger.info(f"Анизотропен звуков хоризонт: r_s({z_star},{theta:.2f},{phi:.2f}) = {r_s:.3f} Mpc")
            return r_s
            
        except Exception as e:
            logger.error(f"Грешка в анизотропния звуков хоризонт: {e}")
            return 147.0  # Fallback към Planck стойност
    
    def directional_volume_averaged_distance(self, z: np.ndarray, theta: float, phi: float) -> np.ndarray:
        """
        Посочно зависимо обемно усреднено разстояние
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            D_V(z,θ,φ) в Mpc
        """
        z = np.asarray(z)
        
        # Анизотропно ъглово разстояние
        D_A = self.anisotropic_angular_diameter_distance(z, theta, phi)
        
        # Анизотропна Хъбъл функция
        H_z = self.anisotropic_hubble_function(z, theta, phi)
        
        # Обемно усреднено разстояние
        factor1 = (1 + z)**2 * D_A**2
        factor2 = c * z / (H_z * 1000)
        
        D_V = (factor1 * factor2)**(1/3)
        
        return D_V
    
    def sky_averaged_quantities(self, z: np.ndarray, n_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Небесно усреднени количества (интегриране по всички посоки)
        
        Args:
            z: Червено отместване
            n_points: Брой точки за интегриране по сферата
            
        Returns:
            Речник с усреднени количества
        """
        z = np.asarray(z)
        
        # Генериране на точки по сферата (Monte Carlo)
        theta_points = np.random.uniform(0, np.pi, n_points)
        phi_points = np.random.uniform(0, 2*np.pi, n_points)
        
        # Инициализация на резултатите
        H_avg = np.zeros_like(z)
        D_A_avg = np.zeros_like(z)
        D_V_avg = np.zeros_like(z)
        
        # Интегриране по сферата
        for theta, phi in zip(theta_points, phi_points):
            sin_theta = np.sin(theta)  # Якобиан за сферични координати
            
            H_dir = self.anisotropic_hubble_function(z, theta, phi)
            D_A_dir = self.anisotropic_angular_diameter_distance(z, theta, phi)
            D_V_dir = self.directional_volume_averaged_distance(z, theta, phi)
            
            H_avg += H_dir * sin_theta
            D_A_avg += D_A_dir * sin_theta
            D_V_avg += D_V_dir * sin_theta
        
        # Нормализация (интегралът от sin(θ) по сферата е 4π)
        normalization = 4 * np.pi / n_points
        
        H_avg *= normalization
        D_A_avg *= normalization  
        D_V_avg *= normalization
        
        return {
            'H_avg': H_avg,
            'D_A_avg': D_A_avg,
            'D_V_avg': D_V_avg,
            'theta_points': theta_points,
            'phi_points': phi_points
        }
    
    def anisotropy_diagnostics(self) -> Dict[str, float]:
        """
        Диагностики на анизотропията
        
        Returns:
            Речник с диагностични параметри
        """
        
        # Общ анизотропен параметър
        total_anisotropy = np.sqrt(self.epsilon_x**2 + self.epsilon_y**2 + self.epsilon_z**2)
        
        # Общо времево забавяне
        total_delay = np.sqrt(self.tau_x**2 + self.tau_y**2 + self.tau_z**2)
        
        # Предпочитана посока (в декартови координати)
        pref_x = np.sin(self.theta_pref) * np.cos(self.phi_pref)
        pref_y = np.sin(self.theta_pref) * np.sin(self.phi_pref)
        pref_z = np.cos(self.theta_pref)
        
        return {
            'total_anisotropy': total_anisotropy,
            'total_delay': total_delay,
            'epsilon_max': max(self.epsilon_x, self.epsilon_y, self.epsilon_z),
            'tau_max': max(self.tau_x, self.tau_y, self.tau_z),
            'angular_strength': self.angular_strength,
            'preferred_direction': (pref_x, pref_y, pref_z),
            'theta_pref_deg': np.degrees(self.theta_pref),
            'phi_pref_deg': np.degrees(self.phi_pref)
        }


def test_anisotropic_cosmology():
    """Тест на анизотропната космология"""
    
    print("🧪 ТЕСТ НА АНИЗОТРОПНА НЕЛИНЕЙНА ВРЕМЕВА КОСМОЛОГИЯ")
    print("=" * 80)
    
    # Създаване на анизотропен модел
    cosmo = AnisotropicNonlinearTimeCosmology(
        # Умерена анизотропия
        epsilon_x=0.05, epsilon_y=0.03, epsilon_z=0.02,
        tau_x=0.08, tau_y=0.05, tau_z=0.02,
        # Предпочитана посока (45° полярно, 30° азимутно)
        theta_preference=np.pi/4, phi_preference=np.pi/6,
        angular_strength=0.5
    )
    
    # Тестови червени отмествания
    z_test = np.array([0.1, 0.5, 1.0, 2.0])
    
    # Тестови посоки
    directions = [
        (0, 0, "Север (z-ос)"),
        (np.pi/2, 0, "Изток (x-ос)"),  
        (np.pi/2, np.pi/2, "Север (y-ос)"),
        (np.pi/4, np.pi/6, "Предпочитана посока")
    ]
    
    print("📊 СРАВНЕНИЕ ПО ПОСОКИ:")
    print("-" * 60)
    
    for theta, phi, name in directions:
        print(f"\n🔍 {name} (θ={np.degrees(theta):.1f}°, φ={np.degrees(phi):.1f}°):")
        
        H_values = cosmo.anisotropic_hubble_function(z_test, theta, phi)
        D_A_values = cosmo.anisotropic_angular_diameter_distance(z_test, theta, phi)
        
        for i, z in enumerate(z_test):
            print(f"  z={z:.1f}: H={H_values[i]:.1f} km/s/Mpc, D_A={D_A_values[i]:.1f} Mpc")
    
    # Небесно усреднени количества
    print(f"\n🌌 НЕБЕСНО УСРЕДНЕНИ КОЛИЧЕСТВА:")
    print("-" * 40)
    
    sky_avg = cosmo.sky_averaged_quantities(z_test, n_points=50)
    
    for i, z in enumerate(z_test):
        print(f"z={z:.1f}: <H>={sky_avg['H_avg'][i]:.1f} km/s/Mpc, <D_A>={sky_avg['D_A_avg'][i]:.1f} Mpc")
    
    # Диагностики
    print(f"\n⚙️ АНИЗОТРОПНИ ДИАГНОСТИКИ:")
    print("-" * 30)
    
    diagnostics = cosmo.anisotropy_diagnostics()
    
    print(f"Обща анизотропия: {diagnostics['total_anisotropy']:.3f}")
    print(f"Общо забавяне: {diagnostics['total_delay']:.3f}")
    print(f"Максимална анизотропия: {diagnostics['epsilon_max']:.3f}")
    print(f"Предпочитана посока: θ={diagnostics['theta_pref_deg']:.1f}°, φ={diagnostics['phi_pref_deg']:.1f}°")
    
    print("\n✅ Тестът завърши успешно!")
    

if __name__ == "__main__":
    test_anisotropic_cosmology() 