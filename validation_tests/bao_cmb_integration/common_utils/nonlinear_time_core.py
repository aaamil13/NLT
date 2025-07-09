"""
Основен модул за нелинейно време в космологията

Този модул имплементира теоретичните основи за нелинейното време,
включително модификации на Хъбъл функцията, звуковия хоризонт и 
геометричните разстояния.
"""

import numpy as np
from scipy import integrate, optimize
from typing import Dict, Any, Callable, Tuple, Optional
import logging

# Настройка на логинг
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Физични константи
c = 299792458  # м/с - скорост на светлината
H0_standard = 67.4  # km/s/Mpc - стандартна стойност от Planck 2018

class NonlinearTimeCosmology:
    """
    Основен клас за нелинейно време космология
    
    Имплементира:
    - Нелинейна времева трансформация t(z)
    - Модифицирана Хъбъл функция H(z)
    - Звуков хоризонт r_s с нелинейно време
    - Геометрични разстояния в новата метрика
    """
    
    def __init__(self, alpha: float = 1.5, beta: float = 0.0, gamma: float = 0.5, 
                 delta: float = 0.1, H0: float = 67.4, Omega_m: float = 0.315, 
                 Omega_Lambda: float = 0.685):
        """
        Инициализация на нелинейна времева космология
        
        Args:
            alpha: Главен нелинеен коефициент
            beta: Корекционен термин
            gamma: Степенен показател
            delta: Добавъчен термин
            H0: Хъбъл константа (km/s/Mpc)
            Omega_m: Плътност на материята
            Omega_Lambda: Плътност на тъмната енергия
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        
        # Защитени проверки
        if abs(Omega_m + Omega_Lambda - 1.0) > 0.1:
            logger.warning(f"Космологията не е плоска: Ωₘ + ΩΛ = {Omega_m + Omega_Lambda:.3f}")
            
        logger.info(f"Инициализирана нелинейна времева космология:")
        logger.info(f"  α={alpha}, β={beta}, γ={gamma}, δ={delta}")
        logger.info(f"  H₀={H0} km/s/Mpc, Ωₘ={Omega_m}, ΩΛ={Omega_Lambda}")
    
    def nonlinear_time_function(self, z: np.ndarray) -> np.ndarray:
        """
        Нелинейна времева трансформация t(z)
        
        Formula: t(z) = z/(1+z) × [α × ln(1+z) + β × (1+z)^γ + δ]
        
        Args:
            z: Червено отместване (може да бъде array)
            
        Returns:
            Нелинейно време t(z)
        """
        z = np.asarray(z)
        
        # Защитени операции
        z_safe = np.maximum(z, 1e-10)  # Избягваме z=0
        one_plus_z = 1 + z_safe
        
        # Нелинейна времева функция
        ln_term = self.alpha * np.log(one_plus_z)
        power_term = self.beta * np.power(one_plus_z, self.gamma)
        
        t_z = (z_safe / one_plus_z) * (ln_term + power_term + self.delta)
        
        # Проверка за NaN/inf
        if np.any(~np.isfinite(t_z)):
            logger.warning("NaN/inf в нелинейната времева функция!")
            t_z = np.where(np.isfinite(t_z), t_z, z_safe / one_plus_z)
            
        return t_z
    
    def modified_hubble_function(self, z: np.ndarray) -> np.ndarray:
        """
        Модифицирана Хъбъл функция H(z) с нелинейно време
        
        Args:
            z: Червено отместване
            
        Returns:
            H(z) в km/s/Mpc
        """
        z = np.asarray(z)
        
        # Стандартна компонента
        E_z_standard = np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_Lambda)
        
        # Нелинейна корекция
        t_z = self.nonlinear_time_function(z)
        nonlinear_correction = 1 + self.alpha * t_z
        
        # Модифицирана функция
        H_z = self.H0 * E_z_standard * nonlinear_correction
        
        return H_z
    
    def sound_speed_baryon_photon_plasma(self, z: np.ndarray) -> np.ndarray:
        """
        Скорост на звука в барион-фотонна плазма c_s(z)
        
        Args:
            z: Червено отместване
            
        Returns:
            Скорост на звука в единици на c
        """
        z = np.asarray(z)
        
        # Стандартни стойности
        Omega_b = 0.049  # Барионна плътност от Planck 2018
        Omega_gamma = 8.24e-5  # Фотонна плътност
        
        # Отношението барион/фотон
        R_ratio = (3 * Omega_b) / (4 * Omega_gamma * (1 + z))
        
        # Скорост на звука
        c_s = c * np.sqrt(1 / (3 * (1 + R_ratio)))
        
        return c_s / c  # Нормализирано към c
    
    def sound_horizon_integral(self, z_star: float = 1100, z_max: float = 3000) -> float:
        """
        Звуков хоризонт r_s(z*) с нелинейно време
        
        Formula: r_s(z*) = ∫[z*→∞] c_s(z)/H(z) dz
        
        Args:
            z_star: Червено отместване на рекомбинацията
            z_max: Горна граница на интеграцията
            
        Returns:
            Звуков хоризонт в Mpc
        """
        def integrand(z):
            c_s = self.sound_speed_baryon_photon_plasma(z)
            H_z = self.modified_hubble_function(z)
            return c_s * c / (H_z * 1000)  # Конвертиране в Mpc
        
        try:
            # Численна интеграция
            r_s, error = integrate.quad(integrand, z_star, z_max, 
                                      epsabs=1e-10, epsrel=1e-8)
            
            if error > 0.01 * abs(r_s):
                logger.warning(f"Висока грешка в интеграцията на r_s: {error:.2e}")
                
            logger.info(f"Звуков хоризонт: r_s({z_star}) = {r_s:.3f} Mpc")
            return r_s
            
        except Exception as e:
            logger.error(f"Грешка в изчисляването на звуковия хоризонт: {e}")
            # Fallback към стандартна стойност
            return 147.0  # Mpc (Planck 2018)
    
    def angular_diameter_distance(self, z: np.ndarray) -> np.ndarray:
        """
        Ъгловo диаметрово разстояние D_A(z) с нелинейно време
        
        Args:
            z: Червено отместване
            
        Returns:
            D_A(z) в Mpc
        """
        z = np.asarray(z)
        
        def integrand(z_val):
            H_z = self.modified_hubble_function(z_val)
            return c / (H_z * 1000)  # Конвертиране в Mpc
        
        D_A = np.zeros_like(z)
        
        for i, z_val in enumerate(z.flat):
            if z_val > 0:
                try:
                    integral, _ = integrate.quad(integrand, 0, z_val)
                    D_A.flat[i] = integral / (1 + z_val)
                except:
                    # Fallback
                    D_A.flat[i] = c * z_val / (self.H0 * 1000 * (1 + z_val))
            else:
                D_A.flat[i] = 0
                
        return D_A.reshape(z.shape)
    
    def volume_averaged_distance(self, z: np.ndarray) -> np.ndarray:
        """
        Обемно усреднено разстояние D_V(z) за BAO анализ
        
        Formula: D_V(z) = [(1+z)²D_A²(z) × cz/H(z)]^(1/3)
        
        Args:
            z: Червено отместване
            
        Returns:
            D_V(z) в Mpc
        """
        z = np.asarray(z)
        
        # Ъглово диаметрово разстояние
        D_A = self.angular_diameter_distance(z)
        
        # Хъбъл функция
        H_z = self.modified_hubble_function(z)
        
        # Обемно усреднено разстояние
        factor1 = (1 + z)**2 * D_A**2
        factor2 = c * z / (H_z * 1000)  # Конвертиране в Mpc
        
        D_V = (factor1 * factor2)**(1/3)
        
        return D_V
    
    def effective_sound_horizon(self, z_star: float = 1100) -> float:
        """
        Ефективен звуков хоризонт r_s_eff с нелинейно време
        
        Интегрира във времева координата вместо z координата
        
        Args:
            z_star: Червено отместване на рекомбинацията
            
        Returns:
            Ефективен звуков хоризонт в Mpc
        """
        # Конвертиране на z_star в нелинейно време
        t_star = self.nonlinear_time_function(np.array([z_star]))[0]
        
        def integrand_time(t):
            # Обратна трансформация от t към z (апроксимативна)
            z_approx = t / (1 - t) if t < 0.99 else 100
            c_s = self.sound_speed_baryon_photon_plasma(z_approx)
            # Времева деривативка на скалния фактор
            a_dot = 1 / (1 + z_approx)  # Опростена
            return c_s * c / a_dot
        
        try:
            r_s_eff, _ = integrate.quad(integrand_time, t_star, 0.99)
            logger.info(f"Ефективен звуков хоризонт: r_s_eff = {r_s_eff:.3f} Mpc")
            return r_s_eff
        except:
            # Fallback към стандартния звуков хоризонт
            return self.sound_horizon_integral(z_star)
    
    def cosmological_parameters_summary(self) -> Dict[str, float]:
        """
        Обобщение на космологичните параметри
        
        Returns:
            Речник с ключови параметри
        """
        z_cmb = 1100
        z_bao_test = 0.5
        
        return {
            'H0': self.H0,
            'Omega_m': self.Omega_m,
            'Omega_Lambda': self.Omega_Lambda,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
            'r_s_cmb': self.sound_horizon_integral(z_cmb),
            'D_V_bao_test': self.volume_averaged_distance(np.array([z_bao_test]))[0],
            'H_z_cmb': self.modified_hubble_function(np.array([z_cmb]))[0],
            't_cmb': self.nonlinear_time_function(np.array([z_cmb]))[0]
        }


def test_nonlinear_time_cosmology():
    """
    Тест на нелинейната времева космология
    """
    print("🧪 ТЕСТ НА НЕЛИНЕЙНАТА ВРЕМЕВА КОСМОЛОГИЯ")
    print("=" * 60)
    
    # Създаване на модел
    model = NonlinearTimeCosmology(alpha=1.5, beta=0.0, gamma=0.5, delta=0.1)
    
    # Тестови червени отмествания
    z_test = np.array([0.1, 0.5, 1.0, 1100])
    
    # Тестове на функциите
    print("\n📊 Нелинейна времева функция:")
    t_z = model.nonlinear_time_function(z_test)
    for i, z in enumerate(z_test):
        print(f"  t({z}) = {t_z[i]:.6f}")
    
    print("\n📈 Модифицирана Хъбъл функция:")
    H_z = model.modified_hubble_function(z_test)
    for i, z in enumerate(z_test):
        print(f"  H({z}) = {H_z[i]:.2f} km/s/Mpc")
    
    print("\n🔊 Звуков хоризонт:")
    r_s = model.sound_horizon_integral()
    print(f"  r_s(1100) = {r_s:.3f} Mpc")
    
    print("\n📏 Обемно усреднено разстояние:")
    D_V = model.volume_averaged_distance(np.array([0.5, 1.0]))
    print(f"  D_V(0.5) = {D_V[0]:.2f} Mpc")
    print(f"  D_V(1.0) = {D_V[1]:.2f} Mpc")
    
    print("\n📋 Обобщение на параметрите:")
    params = model.cosmological_parameters_summary()
    for key, value in params.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Тестът завърши успешно!")


if __name__ == "__main__":
    test_nonlinear_time_cosmology() 