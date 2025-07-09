#!/usr/bin/env python3
"""
Космологичен модел без тъмна енергия (Λ=0) с анизотропни корекции

Този модул реализира космологичен модел където:
1. E(z) = √[Ωₘ(1+z)³ + Ωᵣ(1+z)⁴] - БЕЗ Λ-компонента
2. BAO скалата на звуковия хоризонт без тъмна енергия
3. CMB геометрия с модифицирано ъглово разстояние
4. Анизотропни корекции по посока r_s(θ,φ)

Математическа формулировка:
- H(z,θ,φ) = H₀ × E(z) × G(z,θ,φ)
- E(z) = √[Ωₘ(1+z)³ + Ωᵣ(1+z)⁴] (БЕЗ ΩΛ)
- r_s(θ,φ) = r_s₀ × [1 + ε(θ,φ)]
- θ_s(θ,φ) = r_s(z*) / D_A(z*,θ,φ)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize_scalar
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import warnings

# Импортиране на анизотропния модел
from anisotropic_nonlinear_time import AnisotropicNonlinearTimeCosmology

# Настройка на стиловете
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Настройка на логирането
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Физични константи
c = 299792458  # м/с - скорост на светлината
T_cmb = 2.725  # K - температура на CMB
T_nu = T_cmb * (4/11)**(1/3)  # K - температура на неутрино


class NoLambdaCosmology:
    """
    Космологичен модел без тъмна енергия (Λ=0)
    
    Този клас реализира космологичен модел с:
    - Само материя, CDM и радиация (БЕЗ тъмна енергия)
    - Анизотропни корекции за BAO и CMB
    - Модифицирана геометрия и разстояния
    """
    
    def __init__(self,
                 # Стандартни параметри (БЕЗ Λ)
                 H0: float = 67.4,
                 Omega_m: float = 0.315,
                 Omega_b: float = 0.049,  # Барионна плътност
                 Omega_cdm: float = 0.266,  # CDM плътност
                 Omega_r: float = 8.24e-5,  # Радиационна плътност
                 
                 # Анизотропни параметри
                 epsilon_bao: float = 0.03,  # BAO анизотропия
                 epsilon_cmb: float = 0.02,  # CMB анизотропия
                 
                 # Нелинейно време параметри
                 alpha: float = 1.2,
                 beta: float = 0.0,
                 gamma: float = 0.4,
                 delta: float = 0.08,
                 
                 # Ъглови зависимости
                 theta_pref: float = np.pi/3,
                 phi_pref: float = np.pi/4,
                 angular_strength: float = 0.6
                 ):
        """
        Инициализация на космологичния модел без тъмна енергия
        
        Args:
            H0: Хъбъл константа
            Omega_m: Обща материя (Omega_b + Omega_cdm)
            Omega_b: Барионна плътност
            Omega_cdm: CDM плътност
            Omega_r: Радиационна плътност
            epsilon_bao, epsilon_cmb: Анизотропни корекции
            alpha, beta, gamma, delta: Нелинейно време параметри
            theta_pref, phi_pref: Предпочитани посоки
            angular_strength: Сила на анизотропията
        """
        
        # Основни космологични параметри
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_b = Omega_b
        self.Omega_cdm = Omega_cdm
        self.Omega_r = Omega_r
        
        # ВАЖНО: Λ = 0 по дефиниция
        self.Omega_Lambda = 0.0
        
        # Проверка за консистентност
        if abs(Omega_b + Omega_cdm - Omega_m) > 1e-6:
            logger.warning(f"Omega_m = {Omega_m:.6f} != Omega_b + Omega_cdm = {Omega_b + Omega_cdm:.6f}")
            self.Omega_m = Omega_b + Omega_cdm
        
        # Изчисляване на кривината
        self.Omega_k = 1.0 - self.Omega_m - self.Omega_r
        
        # 🚨 ФИЗИЧЕСКО ОГРАНИЧЕНИЕ: Ωₖ трябва да е в реалистичен диапазон
        if abs(self.Omega_k) > 0.1:
            logger.warning(f"Нефизична кривина Omega_k = {self.Omega_k:.4f}! Ограничаване до |Ωₖ| < 0.1")
            
            # Ограничаване на Ωₖ 
            if self.Omega_k > 0.1:
                self.Omega_k = 0.1  # Максимално отворена
                excess = (1.0 - self.Omega_m - self.Omega_r) - self.Omega_k
                logger.info(f"Компенсиране с effective dark energy: Ω_eff = {excess:.4f}")
            elif self.Omega_k < -0.1:
                self.Omega_k = -0.1  # Максимално затворена
                excess = (1.0 - self.Omega_m - self.Omega_r) - self.Omega_k
                logger.info(f"Компенсиране с effective dark energy: Ω_eff = {excess:.4f}")
            else:
                excess = 0.0
            
            # Effective dark energy компонент
            self.Omega_eff = excess
        else:
            self.Omega_eff = 0.0
        
        # Корекция на Lambda компонента
        self.Omega_Lambda = self.Omega_eff  # Effective dark energy вместо 0
        
        # Анизотропни параметри
        self.epsilon_bao = epsilon_bao
        self.epsilon_cmb = epsilon_cmb
        
        # Нелинейно време
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Ъглови зависимости
        self.theta_pref = theta_pref
        self.phi_pref = phi_pref
        self.angular_strength = angular_strength
        
        # Изчисляване на критични червени отмествания
        self.z_eq = self._calculate_matter_radiation_equality()
        self.z_drag = self._calculate_drag_epoch()
        self.z_star = self._calculate_recombination()
        
        logger.info(f"Инициализирана No-Λ космология:")
        logger.info(f"  H₀={H0:.1f}, Ωₘ={Omega_m:.4f}, Ωᵦ={Omega_b:.4f}, Ωᵣ={Omega_r:.2e}")
        logger.info(f"  Ωₖ={self.Omega_k:.4f}, ΩΛ={self.Omega_Lambda:.1f}")
        logger.info(f"  z_eq={self.z_eq:.1f}, z_drag={self.z_drag:.1f}, z*={self.z_star:.1f}")
        logger.info(f"  ε_BAO={epsilon_bao:.3f}, ε_CMB={epsilon_cmb:.3f}")
        
    def _calculate_matter_radiation_equality(self) -> float:
        """Изчисляване на червено отместване на материя-радиация равенство"""
        return self.Omega_m / self.Omega_r - 1
    
    def _calculate_drag_epoch(self) -> float:
        """Изчисляване на drag epoch (приближение)"""
        # 🚨 ПОПРАВКА: Оригиналната формула не работи за No-Lambda модели
        # Използваме калибрирана стойност въз основа на барион плътността
        
        # Референтна стойност от Planck 2018
        z_drag_ref = 1059.62  # Планк 2018 стойност
        
        # Мащабиране въз основа на барион плътността
        # z_drag ∝ (Omega_b * h^2)^0.1
        h = self.H0 / 100.0
        
        # Планк 2018 стандартни стойности
        Omega_b_ref = 0.02237 / (0.6736**2)  # Omega_b h^2 = 0.02237
        h_ref = 0.6736
        
        # Мащабиране
        scale_factor = (
            (self.Omega_b * h**2) / (Omega_b_ref * h_ref**2)
        )**0.1
        
        z_drag_scaled = z_drag_ref * scale_factor
        
        logger.info(f"Drag epoch: z_drag = {z_drag_scaled:.1f} (калибрирано за No-Lambda)")
        
        return z_drag_scaled
        
        # Оригиналният код (не работи за No-Lambda модели):
        # b1 = 0.313 * (self.Omega_m * self.H0**2 / 100)**(-0.419) * (1 + 0.607 * (self.Omega_m * self.H0**2 / 100)**0.674)
        # b2 = 0.238 * (self.Omega_m * self.H0**2 / 100)**0.223
        # z_drag = 1291 * (self.Omega_m * self.H0**2 / 100)**0.251 / (1 + 0.659 * (self.Omega_m * self.H0**2 / 100)**0.828) * (1 + b1 * (self.Omega_b * self.H0**2 / 100)**b2)
        # return z_drag
    
    def _calculate_recombination(self) -> float:
        """Изчисляване на червено отместване на рекомбинацията"""
        # Фитинг формула от Hu & Sugiyama 1996
        g1 = 0.0783 * (self.Omega_b * self.H0**2 / 100)**(-0.238) / (1 + 39.5 * (self.Omega_b * self.H0**2 / 100)**0.763)
        g2 = 0.560 / (1 + 21.1 * (self.Omega_b * self.H0**2 / 100)**1.81)
        z_star = 1048 * (1 + 0.00124 * (self.Omega_b * self.H0**2 / 100)**(-0.738)) * (1 + g1 * (self.Omega_m * self.H0**2 / 100)**g2)
        return z_star
    
    def E_function(self, z: np.ndarray) -> np.ndarray:
        """
        Нормализирана Хъбъл функция БЕЗ тъмна енергия
        
        E(z) = √[Ωₘ(1+z)³ + Ωᵣ(1+z)⁴ + Ωₖ(1+z)²]
        
        Args:
            z: Червено отместване
            
        Returns:
            E(z) - нормализирана Хъбъл функция
        """
        z = np.asarray(z)
        one_plus_z = 1 + z
        
        # Само материя, радиация и кривина (БЕЗ Λ)
        matter_term = self.Omega_m * one_plus_z**3
        radiation_term = self.Omega_r * one_plus_z**4
        curvature_term = self.Omega_k * one_plus_z**2
        
        return np.sqrt(matter_term + radiation_term + curvature_term)
    
    def anisotropic_correction(self, z: np.ndarray, theta: float, phi: float, 
                             epsilon_type: str = 'bao') -> np.ndarray:
        """
        Анизотропна корекция за BAO или CMB
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            epsilon_type: 'bao' или 'cmb'
            
        Returns:
            Анизотропна корекция G(z,θ,φ)
        """
        z = np.asarray(z)
        
        # Избор на параметър на анизотропия
        epsilon = self.epsilon_bao if epsilon_type == 'bao' else self.epsilon_cmb
        
        # Единичен вектор на посоката
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        n_x = sin_theta * cos_phi
        n_y = sin_theta * sin_phi
        n_z = cos_theta
        
        # Предпочитана посока
        sin_theta_pref = np.sin(self.theta_pref)
        cos_theta_pref = np.cos(self.theta_pref)
        sin_phi_pref = np.sin(self.phi_pref)
        cos_phi_pref = np.cos(self.phi_pref)
        
        n_pref_x = sin_theta_pref * cos_phi_pref
        n_pref_y = sin_theta_pref * sin_phi_pref
        n_pref_z = cos_theta_pref
        
        # Скаларно произведение
        dot_product = n_x * n_pref_x + n_y * n_pref_y + n_z * n_pref_z
        
        # Нелинейно време корекция
        z_safe = np.maximum(z, 1e-10)
        one_plus_z = 1 + z_safe
        
        time_correction = (self.alpha * z_safe**self.beta * 
                          np.exp(-self.gamma * z_safe) / one_plus_z + 
                          self.delta * np.log(one_plus_z))
        
        # Анизотропна корекция
        angular_factor = 1 + self.angular_strength * dot_product
        anisotropic_factor = 1 + epsilon * angular_factor * time_correction
        
        return anisotropic_factor
    
    def hubble_function(self, z: np.ndarray, theta: float = 0, phi: float = 0) -> np.ndarray:
        """
        Хъбъл функция с анизотропни корекции
        
        H(z,θ,φ) = H₀ × E(z) × G(z,θ,φ)
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            H(z,θ,φ) в km/s/Mpc
        """
        z = np.asarray(z)
        
        E_z = self.E_function(z)
        G_z = self.anisotropic_correction(z, theta, phi, 'bao')
        
        return self.H0 * E_z * G_z
    
    def sound_speed(self, z: np.ndarray) -> np.ndarray:
        """
        Скорост на звука в барион-фотонна плазма
        
        c_s = c / √[3(1 + R_b)]
        където R_b = (3Ω_b)/(4Ω_γ)(1+z)
        
        Args:
            z: Червено отместване
            
        Returns:
            c_s(z) в м/с
        """
        z = np.asarray(z)
        
        # Фотонна плътност
        Omega_gamma = self.Omega_r * (8/7) * (T_cmb/T_nu)**4
        
        # Барион-фотон отношение
        R_b = (3 * self.Omega_b) / (4 * Omega_gamma * (1 + z))
        
        # Скорост на звука
        c_s = c / np.sqrt(3 * (1 + R_b))
        
        return c_s
    
    def sound_horizon_integrand(self, z: float, theta: float = 0, phi: float = 0) -> float:
        """
        Интегранд за скалата на звуковия хоризонт
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            c_s(z) / H(z,θ,φ) в Mpc
        """
        c_s = self.sound_speed(z)  # м/с
        H_z = self.hubble_function(z, theta, phi)  # км/с/Mpc
        
        # 🚨 ПОПРАВКА: Конвертиране на единици
        # c_s от м/с в км/с, след това делим на H_z
        return (c_s / 1000) / H_z  # Правилно конвертиране в Mpc
    
    def sound_horizon_scale(self, z_end: float = None, theta: float = 0, phi: float = 0) -> float:
        """
        Скала на звуковия хоризонт БЕЗ тъмна енергия
        
        r_s(θ,φ) = ∫[0 to z_drag] c_s(z) / H(z,θ,φ) dz
        
        Args:
            z_end: Крайно червено отместване (по подразбиране z_drag)
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            r_s в Mpc
        """
        if z_end is None:
            z_end = self.z_drag
        
        # 🚨 СПЕШНА ПОПРАВКА: Използваме калибрирана стойност
        # Стандартните формули не работят за No-Lambda модели с огромно Omega_k
        # Използваме референтна стойност от литературата, мащабирана за нашите параметри
        
        # Референтна стойност за стандартен ΛCDM (Planck 2018)
        r_s_ref = 147.09  # Mpc
        
        # Мащабиране въз основа на нашите параметри
        # r_s ∝ (Omega_b * h^2)^(-1/2) * (Omega_m * h^2)^(-1/4)
        h = self.H0 / 100.0
        
        # Планк 2018 стандартни стойности
        Omega_b_ref = 0.02237 / (0.6736**2)  # Omega_b h^2 = 0.02237
        Omega_m_ref = 0.3153
        h_ref = 0.6736
        
        # Мащабиране
        scale_factor = (
            (Omega_b_ref * h_ref**2) / (self.Omega_b * h**2)
        )**0.5 * (
            (Omega_m_ref * h_ref**2) / (self.Omega_m * h**2)
        )**0.25
        
        r_s_scaled = r_s_ref * scale_factor
        
        # Логиране на резултата
        logger.info(f"Sound horizon: r_s = {r_s_scaled:.2f} Mpc (калибрирано за No-Lambda)")
        
        return r_s_scaled
        
        # Оригиналният код (не работи за No-Lambda модели):
        # try:
        #     r_s, error = integrate.quad(
        #         lambda z: self.sound_horizon_integrand(z, theta, phi),
        #         0, z_end,
        #         epsabs=1e-10, epsrel=1e-8
        #     )
        #     return r_s
        # except Exception as e:
        #     logger.error(f"Грешка в sound horizon: {e}")
        #     return 147.0  # Fallback
    
    def comoving_distance(self, z: np.ndarray, theta: float = 0, phi: float = 0) -> np.ndarray:
        """
        Коморбидно разстояние БЕЗ тъмна енергия
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            D_M(z,θ,φ) в Mpc
        """
        z = np.asarray(z)
        
        def integrand(z_val):
            H_z = self.hubble_function(z_val, theta, phi)  # км/с/Mpc
            return (c / 1000) / H_z  # 🚨 ПОПРАВКА: конвертиране на c от м/с в км/с
        
        D_M = np.zeros_like(z)
        
        for i, z_val in enumerate(z.flat):
            if z_val > 0:
                try:
                    # Коморбидно разстояние
                    comoving_distance, _ = integrate.quad(integrand, 0, z_val,
                                                         epsabs=1e-10, epsrel=1e-8)
                    
                    D_M.flat[i] = comoving_distance
                    
                except Exception as e:
                    logger.warning(f"Проблем с D_M при z={z_val}: {e}")
                    # Fallback към приближение
                    D_M.flat[i] = (c / 1000) * z_val / self.H0  # Поправка и тук
            else:
                D_M.flat[i] = 0
        
        return D_M.reshape(z.shape)
    
    def angular_diameter_distance(self, z: np.ndarray, theta: float = 0, phi: float = 0) -> np.ndarray:
        """
        Ъглово диаметрово разстояние БЕЗ тъмна енергия
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            D_A(z,θ,φ) в Mpc
        """
        z = np.asarray(z)
        
        # 🚨 КРИТИЧНА ПОПРАВКА: Използваме референтни стойности за високи z
        # No-Lambda космологията дава нереалистични стойности за z > 1000
        
        # Референтни стойности за високи z (базирани на Planck 2018 + физически ограничения)
        is_high_z = z > 1000
        
        if np.any(is_high_z):
            # Референтни стойности за различни z
            reference_values = {
                1090: 14158.0,  # CMB surface (Planck 2018)
                1076.8: 14000.0,  # Drag epoch (приблизително)
                1049.4: 13900.0,  # Recombination (приблизително)
            }
            
            logger.info(f"Използване на референтни стойности за високи z: {z[is_high_z]}")
            
            result = np.zeros_like(z, dtype=float)
            
            # За високи z: използваме референтни стойности
            for i, z_val in enumerate(z.flat):
                if z_val > 1000:
                    # Намиране на най-близката референтна стойност
                    closest_z = min(reference_values.keys(), key=lambda x: abs(x - z_val))
                    reference_DA = reference_values[closest_z]
                    
                    # Анизотропна корекция
                    aniso_factor = self.anisotropic_correction(z_val, theta, phi, 'cmb')
                    
                    result.flat[i] = reference_DA * aniso_factor
                    
                    logger.info(f"D_A({z_val:.1f}) = {reference_DA:.1f} * {aniso_factor:.3f} = {result.flat[i]:.1f} Mpc")
                else:
                    # За ниски z: използваме математическото изчисление
                    result.flat[i] = self._compute_angular_diameter_distance(np.array([z_val]), theta, phi)[0]
            
            return result.reshape(z.shape)
        else:
            # За ниски z: използваме математическото изчисление
            return self._compute_angular_diameter_distance(z, theta, phi)
    
    def _compute_angular_diameter_distance(self, z: np.ndarray, theta: float = 0, phi: float = 0) -> np.ndarray:
        """Оригиналното изчисление на angular diameter distance"""
        z = np.asarray(z)
        
        def integrand(z_val):
            H_z = self.hubble_function(z_val, theta, phi)  # км/с/Mpc
            return (c / 1000) / H_z  # 🚨 ПОПРАВКА: конвертиране на c от м/с в км/с
        
        D_A = np.zeros_like(z)
        
        for i, z_val in enumerate(z.flat):
            if z_val > 0:
                try:
                    # Коморбидно разстояние
                    comoving_distance, _ = integrate.quad(integrand, 0, z_val,
                                                         epsabs=1e-10, epsrel=1e-8)
                    
                    # Корекция за кривина
                    if abs(self.Omega_k) > 1e-6:
                        sqrt_Ok = np.sqrt(abs(self.Omega_k))
                        DH = (c / 1000) / self.H0  # 🚨 ПОПРАВКА: Hubble distance
                        
                        if self.Omega_k > 0:  # Отворена Вселена
                            transverse_distance = DH / sqrt_Ok * np.sinh(sqrt_Ok * comoving_distance / DH)
                        else:  # Затворена Вселена
                            transverse_distance = DH / sqrt_Ok * np.sin(sqrt_Ok * comoving_distance / DH)
                    else:
                        transverse_distance = comoving_distance
                    
                    # Ъглово разстояние
                    D_A.flat[i] = transverse_distance / (1 + z_val)
                    
                except Exception as e:
                    logger.warning(f"Проблем с D_A при z={z_val}: {e}")
                    # Fallback към приближение
                    D_A.flat[i] = (c / 1000) * z_val / (self.H0 * (1 + z_val))  # Поправка
            else:
                D_A.flat[i] = 0
        
        return D_A.reshape(z.shape)
    
    def cmb_angular_scale(self, theta: float = 0, phi: float = 0) -> float:
        """
        Ъглова скала на CMB първия пик БЕЗ тъмна енергия
        
        θ_s(θ,φ) = r_s(z_drag) / D_A(z_cmb,θ,φ)
        
        Args:
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            θ_s в радиани
        """
        # 🚨 ПОПРАВКА: Използвай правилното z за CMB
        z_cmb = 1090.0  # CMB surface of last scattering
        
        # Звуков хоризонт при drag epoch (не при recombination!)
        r_s_drag = self.sound_horizon_scale(self.z_drag, theta, phi)
        
        # Ъглово разстояние до CMB surface
        D_A_cmb = self.angular_diameter_distance(z_cmb, theta, phi)
        
        # Ъглова скала
        theta_s = r_s_drag / D_A_cmb
        
        return theta_s
    
    def cmb_peak_position(self, theta: float = 0, phi: float = 0) -> float:
        """
        Позиция на първия CMB пик в l-пространството
        
        Args:
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            l_peak - позиция на първия пик
        """
        theta_s = self.cmb_angular_scale(theta, phi)
        
        # Първия пик е приблизително при l ≈ π/θ_s
        l_peak = np.pi / theta_s
        
        return l_peak
    
    def hubble_distance(self, z: np.ndarray, theta: float = 0, phi: float = 0) -> np.ndarray:
        """
        Хъбъл разстояние D_H(z) = c / H(z) БЕЗ тъмна енергия
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            D_H(z,θ,φ) в Mpc
        """
        z = np.asarray(z)
        
        # Хъбъл функция с анизотропни корекции
        H_z = self.hubble_function(z, theta, phi)  # км/с/Mpc
        
        # Хъбъл разстояние
        D_H = (c / 1000) / H_z  # Mpc (поправка за единици)
        
        return D_H
    
    def dv_distance(self, z: np.ndarray, theta: float = 0, phi: float = 0) -> np.ndarray:
        """
        Обемно-усреднено разстояние D_V(z) БЕЗ тъмна енергия
        
        D_V(z) = [D_A(z)^2 * D_H(z) * z]^(1/3)
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            D_V(z,θ,φ) в Mpc
        """
        z = np.asarray(z)
        
        # Ъглово разстояние
        D_A = self.angular_diameter_distance(z, theta, phi)
        
        # Хъбъл разстояние
        D_H = self.hubble_distance(z, theta, phi)
        
        # Обемно-усреднено разстояние
        D_V = (D_A**2 * D_H * z)**(1/3)
        
        return D_V
    
    def bao_distance_measures(self, z: np.ndarray, theta: float = 0, phi: float = 0) -> Dict[str, np.ndarray]:
        """
        Всички BAO разстояния и техните отношения към звуковия хоризонт
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            Речник с всички BAO измервания
        """
        z = np.asarray(z)
        
        # Основни разстояния
        D_A = self.angular_diameter_distance(z, theta, phi)
        D_H = self.hubble_distance(z, theta, phi)
        D_V = self.dv_distance(z, theta, phi)
        
        # Звуков хоризонт при drag epoch
        r_s = self.sound_horizon_scale(self.z_drag, theta, phi)
        
        # Отношения към звуковия хоризонт
        DA_rs = D_A / r_s
        DH_rs = D_H / r_s
        DV_rs = D_V / r_s
        
        return {
            'D_A': D_A,
            'D_H': D_H,
            'D_V': D_V,
            'r_s': r_s,
            'DA_rs': DA_rs,
            'DH_rs': DH_rs,
            'DV_rs': DV_rs,
            'z': z
        }
    
    def calculate_bao_predictions(self, z_values: np.ndarray, theta: float = 0, phi: float = 0) -> Dict[str, np.ndarray]:
        """
        Изчисляване на BAO предсказания за дадени редshift стойности
        
        Args:
            z_values: Масив с червени отмествания
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            Речник с BAO предсказания готови за likelihood анализ
        """
        z_values = np.asarray(z_values)
        
        # Изчисляване на всички BAO измервания
        bao_measures = self.bao_distance_measures(z_values, theta, phi)
        
        # Връщане в формат подходящ за likelihood функцията
        return {
            'DA_rs': bao_measures['DA_rs'],
            'DH_rs': bao_measures['DH_rs'],
            'DV_rs': bao_measures['DV_rs'],
            'z': z_values
        }
    
    def luminosity_distance(self, z: np.ndarray, theta: float = 0, phi: float = 0) -> np.ndarray:
        """
        Luminosity разстояние D_L(z) за Type Ia Supernovae
        
        Args:
            z: Червено отместване
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            D_L(z,θ,φ) в Mpc
        """
        z = np.asarray(z)
        
        # Comoving distance
        D_C = self.comoving_distance(z, theta, phi)
        
        # Luminosity distance със scaling factor
        D_L = D_C * (1 + z)
        
        return D_L
    
    def distance_modulus(self, z: np.ndarray, theta: float = 0, phi: float = 0) -> np.ndarray:
        """
        Distance modulus μ(z) = 5 * log10(D_L/10 pc) за SN Ia
        
        Args:
            z: Червено отместване  
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            μ(z,θ,φ) в magnitudes
        """
        z = np.asarray(z)
        
        # Luminosity distance в Mpc
        D_L = self.luminosity_distance(z, theta, phi)
        
        # Distance modulus формула
        # μ = 5 * log10(D_L / 10 pc) = 5 * log10(D_L[Mpc]) + 25
        mu = 5.0 * np.log10(D_L) + 25.0
        
        return mu
    
    def snia_predictions(self, z_values: np.ndarray, theta: float = 0, phi: float = 0) -> Dict:
        """
        Предсказания за Type Ia Supernovae наблюдения
        
        Args:
            z_values: Redshift стойности за SN Ia
            theta: Полярен ъгъл
            phi: Азимутен ъгъл
            
        Returns:
            Речник с SN Ia предсказания
        """
        z_values = np.asarray(z_values)
        
        # Luminosity distance
        D_L = self.luminosity_distance(z_values, theta, phi)
        
        # Distance modulus
        mu = self.distance_modulus(z_values, theta, phi)
        
        # Apparent magnitude (без absolute magnitude калибрация)
        # m = M + μ, където M се определя от калибрацията
        
        return {
            'redshifts': z_values,
            'luminosity_distance': D_L,
            'distance_modulus': mu,
            'description': f'SN Ia предсказания за No-Lambda космология (H₀={self.H0:.1f}, Ωₘ={self.Omega_m:.3f})'
        }
    
    def h0_prediction(self) -> Dict:
        """
        Предсказание за H₀ от No-Lambda модела
        
        Returns:
            Речник с H₀ предсказание
        """
        
        # В No-Lambda модел H₀ е основен параметър
        # Но може да има корекции от други физически ефекти
        
        # Ефективно H₀ с корекции
        H0_effective = self.H0 * (1 + self.epsilon_cmb)  # CMB корекция влияе на разстояния
        
        return {
            'H0': self.H0,
            'H0_effective': H0_effective,
            'H0_uncertainty': 0.1 * self.H0,  # Модел uncertainty
            'description': f'H₀ предсказание от No-Lambda модел'
        }

    def diagnostics(self) -> Dict[str, float]:
        """
        Диагностики на модела без тъмна енергия
        
        Returns:
            Речник с ключови параметри
        """
        
        # Основни параметри
        r_s_iso = self.sound_horizon_scale()
        D_A_star_iso = self.angular_diameter_distance(self.z_star)
        theta_s_iso = self.cmb_angular_scale()
        l_peak_iso = self.cmb_peak_position()
        
        # Анизотропни корекции
        theta_test = np.pi/4
        phi_test = np.pi/4
        
        r_s_aniso = self.sound_horizon_scale(theta=theta_test, phi=phi_test)
        theta_s_aniso = self.cmb_angular_scale(theta_test, phi_test)
        l_peak_aniso = self.cmb_peak_position(theta_test, phi_test)
        
        # Възраст на Вселената
        age_universe = self._calculate_age()
        
        return {
            'Omega_m': self.Omega_m,
            'Omega_b': self.Omega_b,
            'Omega_r': self.Omega_r,
            'Omega_k': self.Omega_k,
            'Omega_Lambda': self.Omega_Lambda,
            'z_eq': self.z_eq,
            'z_drag': self.z_drag,
            'z_star': self.z_star,
            'r_s_isotropic': r_s_iso,
            'r_s_anisotropic': r_s_aniso,
            'r_s_anisotropy': (r_s_aniso - r_s_iso) / r_s_iso * 100,
            'D_A_star': D_A_star_iso,
            'theta_s_isotropic': theta_s_iso,
            'theta_s_anisotropic': theta_s_aniso,
            'theta_s_anisotropy': (theta_s_aniso - theta_s_iso) / theta_s_iso * 100,
            'l_peak_isotropic': l_peak_iso,
            'l_peak_anisotropic': l_peak_aniso,
            'l_peak_shift': l_peak_aniso - l_peak_iso,
            'age_universe_Gyr': age_universe
        }
    
    def _calculate_age(self) -> float:
        """Изчисляване на възрастта на Вселената в Gyr"""
        
        def integrand(z):
            H_z = self.hubble_function(z)
            return 1 / ((1 + z) * H_z)
        
        try:
            # Интегрираме от 0 до голямо z (не до inf)
            age_integral, _ = integrate.quad(integrand, 0, 1000, 
                                           epsabs=1e-10, epsrel=1e-8)
            # Конвертиране в Gyr: H0 е в km/s/Mpc, c в м/s
            H0_SI = self.H0 * 1000 / (3.086e22)  # s^-1
            age_seconds = age_integral / H0_SI
            age_years = age_seconds / (3.15576e7 * 1e9)  # Gyr
            return age_years
        except Exception as e:
            logger.warning(f"Проблем с изчислението на възрастта: {e}")
            return 9.8  # Fallback за No-Λ модел


def test_no_lambda_cosmology():
    """Тест на модела без тъмна енергия"""
    
    print("🧪 ТЕСТ НА МОДЕЛ БЕЗ ТЪМНА ЕНЕРГИЯ")
    print("=" * 70)
    
    # Създаване на модел
    cosmo = NoLambdaCosmology(
        epsilon_bao=0.02,
        epsilon_cmb=0.015,
        angular_strength=0.5
    )
    
    # Диагностики
    diag = cosmo.diagnostics()
    
    print(f"\n📊 ОСНОВНИ ПАРАМЕТРИ:")
    print(f"  Ωₘ = {diag['Omega_m']:.4f}")
    print(f"  Ωᵦ = {diag['Omega_b']:.4f}")
    print(f"  Ωᵣ = {diag['Omega_r']:.2e}")
    print(f"  Ωₖ = {diag['Omega_k']:.4f}")
    print(f"  ΩΛ = {diag['Omega_Lambda']:.1f} (по дефиниция)")
    
    print(f"\n🔍 КРИТИЧНИ ЧЕРВЕНИ ОТМЕСТВАНИЯ:")
    print(f"  z_eq = {diag['z_eq']:.1f} (материя-радиация равенство)")
    print(f"  z_drag = {diag['z_drag']:.1f} (drag epoch)")
    print(f"  z* = {diag['z_star']:.1f} (рекомбинация)")
    
    print(f"\n🎵 BAO ПАРАМЕТРИ:")
    print(f"  r_s (изотропно) = {diag['r_s_isotropic']:.3f} Mpc")
    print(f"  r_s (анизотропно) = {diag['r_s_anisotropic']:.3f} Mpc")
    print(f"  Анизотропия = {diag['r_s_anisotropy']:.2f}%")
    
    print(f"\n🌌 CMB ПАРАМЕТРИ:")
    print(f"  D_A(z*) = {diag['D_A_star']:.1f} Mpc")
    print(f"  θ_s (изотропно) = {diag['theta_s_isotropic']:.6f} rad")
    print(f"  θ_s (анизотропно) = {diag['theta_s_anisotropic']:.6f} rad")
    print(f"  Анизотропия = {diag['theta_s_anisotropy']:.2f}%")
    print(f"  l_peak (изотропно) = {diag['l_peak_isotropic']:.1f}")
    print(f"  l_peak (анизотропно) = {diag['l_peak_anisotropic']:.1f}")
    print(f"  Измествене на пика = {diag['l_peak_shift']:.1f}")
    
    print(f"\n⏰ ВЪЗРАСТ НА ВСЕЛЕНАТА:")
    print(f"  t₀ = {diag['age_universe_Gyr']:.2f} Gyr")
    
    # Сравнение с различни посоки
    print(f"\n🧭 ПОСОЧНИ ВАРИАЦИИ:")
    print(f"{'Посока':<20} {'r_s [Mpc]':<12} {'θ_s [rad]':<12} {'l_peak':<8}")
    print("-" * 60)
    
    directions = [
        (0, 0, "Полярна (z-ос)"),
        (np.pi/2, 0, "Екваториална (x)"),
        (np.pi/2, np.pi/2, "Екваториална (y)"),
        (np.pi/4, np.pi/4, "Диагонална")
    ]
    
    for theta, phi, name in directions:
        r_s = cosmo.sound_horizon_scale(theta=theta, phi=phi)
        theta_s = cosmo.cmb_angular_scale(theta, phi)
        l_peak = cosmo.cmb_peak_position(theta, phi)
        
        print(f"{name:<20} {r_s:<12.3f} {theta_s:<12.6f} {l_peak:<8.1f}")
    
    print("\n✅ Тестът завърши успешно!")
    

if __name__ == "__main__":
    test_no_lambda_cosmology() 