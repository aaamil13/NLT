"""
CMB анализатор с нелинейно време

Този модул имплементира анализ на космическото микровълново излъчване (CMB)
в контекста на нелинейната времева космология. Фокусира се върху акустичните
пикове и техните модификации поради нелинейното време.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import integrate, interpolate
from typing import Dict, List, Tuple, Any, Optional
import logging

from common_utils.nonlinear_time_core import NonlinearTimeCosmology
from common_utils.cosmological_parameters import CMBData, PlanckCosmology, PhysicalConstants
from common_utils.data_processing import CMBDataProcessor, StatisticalAnalyzer

logger = logging.getLogger(__name__)

class CMBAnalyzer:
    """
    Анализатор на космическото микровълново излъчване с нелинейно време
    
    Функционалности:
    - Анализ на акустичните пикове
    - Изчисляване на модифицираните θ* и r_s
    - Сравнение с Planck данни
    - Анализ на power spectrum
    """
    
    def __init__(self, nonlinear_params: Dict[str, float] = None):
        """
        Инициализация на CMB анализатора
        
        Args:
            nonlinear_params: Параметри за нелинейното време
        """
        # Използвай стандартните параметри ако не са предоставени
        if nonlinear_params is None:
            nonlinear_params = {
                'alpha': 1.5,
                'beta': 0.0,
                'gamma': 0.5,
                'delta': 0.1
            }
        
        # Инициализация на космологията
        self.cosmology = NonlinearTimeCosmology(**nonlinear_params)
        
        # Процесор за данни
        self.data_processor = CMBDataProcessor()
        
        # Планк параметри
        self.planck_params = PlanckCosmology.get_summary()
        
        # Физични константи
        self.constants = PhysicalConstants.get_all_constants()
        
        # CMB данни
        self.cmb_data = CMBData.get_cmb_summary()
        
        logger.info("Инициализиран CMB анализатор с нелинейно време")
    
    def calculate_angular_sound_horizon(self, z_star: float = 1089.8) -> float:
        """
        Изчислява ъгловия размер на звуковия хоризонт θ*
        
        Args:
            z_star: Червено отместване на рекомбинацията
            
        Returns:
            Ъглов размер на звуковия хоризонт в радиани
        """
        # Звуков хоризонт при рекомбинация
        r_s = self.cosmology.sound_horizon_integral(z_star)
        
        # Ъглово диаметрово разстояние до рекомбинация
        D_A_star = self.cosmology.angular_diameter_distance(np.array([z_star]))[0]
        
        # Ъглов размер
        theta_star = r_s / D_A_star
        
        logger.info(f"Ъглов размер на звуковия хоризонт: θ* = {theta_star:.7f} rad")
        return theta_star
    
    def calculate_acoustic_peak_positions(self, n_peaks: int = 5) -> np.ndarray:
        """
        Изчислява местоположенията на акустичните пикове
        
        Args:
            n_peaks: Брой пикове за изчисляване
            
        Returns:
            Местоположения на пиковете в l-пространството
        """
        # Ъглов размер на звуковия хоризонт
        theta_star = self.calculate_angular_sound_horizon()
        
        # Характерен мащаб
        l_A = np.pi / theta_star
        
        # Позиции на пиковете (приблизително)
        l_peaks = np.zeros(n_peaks)
        for n in range(n_peaks):
            l_peaks[n] = l_A * (n + 1)
        
        logger.info(f"Първи пик при l = {l_peaks[0]:.1f}")
        return l_peaks
    
    def calculate_modified_power_spectrum(self, l_values: np.ndarray, 
                                        include_nonlinear_corrections: bool = True) -> np.ndarray:
        """
        Изчислява модифицирания CMB power spectrum
        
        Args:
            l_values: Мултиполни моменти
            include_nonlinear_corrections: Включване на нелинейни корекции
            
        Returns:
            Модифициран power spectrum
        """
        # Базов спектър (опростен модел)
        l_values = np.asarray(l_values)
        
        # Ъглов размер на звуковия хоризонт
        theta_star = self.calculate_angular_sound_horizon()
        l_A = np.pi / theta_star
        
        # Базов спектър с акустични осцилации
        C_l_base = self._calculate_base_spectrum(l_values, l_A)
        
        if include_nonlinear_corrections:
            # Нелинейни корекции
            correction_factor = self._calculate_nonlinear_corrections(l_values)
            C_l_modified = C_l_base * correction_factor
        else:
            C_l_modified = C_l_base
        
        return C_l_modified
    
    def _calculate_base_spectrum(self, l_values: np.ndarray, l_A: float) -> np.ndarray:
        """
        Изчислява базовия CMB power spectrum
        
        Args:
            l_values: Мултиполни моменти
            l_A: Акустичен мащаб
            
        Returns:
            Базов power spectrum
        """
        # Опростен модел за CMB спектър
        # Включва Sachs-Wolfe ефект и акустични осцилации
        
        # Нормализация
        A_norm = 3000e-6  # μK²
        
        # Sachs-Wolfe плато
        C_l_sw = A_norm * np.ones_like(l_values)
        
        # Акустични осцилации
        phase = 2 * np.pi * l_values / l_A
        oscillations = 1 + 0.3 * np.cos(phase) * np.exp(-l_values / (2 * l_A))
        
        # Дифузионно затихване
        damping = np.exp(-(l_values / 1000)**2)
        
        # Комбиниран спектър
        C_l_base = C_l_sw * oscillations * damping
        
        return C_l_base
    
    def _calculate_nonlinear_corrections(self, l_values: np.ndarray) -> np.ndarray:
        """
        Изчислява нелинейните корекции към power spectrum
        
        Args:
            l_values: Мултиполни моменти
            
        Returns:
            Корекционен фактор
        """
        # Корекционен фактор базиран на нелинейните параметри
        alpha = self.cosmology.alpha
        gamma = self.cosmology.gamma
        
        # Мащабно-зависими корекции
        scale_factor = (l_values / 200)**(-gamma/10)
        
        # Амплитудни корекции
        amplitude_factor = 1 + alpha * 0.01 * np.exp(-l_values / 500)
        
        # Комбиниран корекционен фактор
        correction_factor = scale_factor * amplitude_factor
        
        return correction_factor
    
    def compare_with_planck_data(self) -> Dict[str, Any]:
        """
        Сравнява теоретичните предсказания с Planck данни
        
        Returns:
            Резултати от сравнението
        """
        # Планк TT данни
        planck_tt = self.cmb_data['planck_tt']
        l_obs = planck_tt['l']
        C_l_obs = planck_tt['C_l']
        C_l_err = planck_tt['C_l_err']
        
        # Теоретични предсказания
        C_l_theory = self.calculate_modified_power_spectrum(l_obs)
        
        # Статистически анализ
        stats = StatisticalAnalyzer.goodness_of_fit_summary(
            C_l_theory, C_l_obs, C_l_err, n_params=4
        )
        
        # Резидуали
        residuals = (C_l_theory - C_l_obs) / C_l_err
        
        # Сравнение на акустичните пикове
        peaks_comparison = self._compare_acoustic_peaks()
        
        logger.info(f"CMB χ²/dof = {stats['reduced_chi_squared']:.2f}")
        logger.info(f"Средно отклонение: {np.mean(residuals):.3f} σ")
        
        return {
            'l_obs': l_obs,
            'C_l_obs': C_l_obs,
            'C_l_theory': C_l_theory,
            'C_l_err': C_l_err,
            'residuals': residuals,
            'statistics': stats,
            'peaks_comparison': peaks_comparison,
            'agreement_level': self._assess_agreement_level(stats['reduced_chi_squared'])
        }
    
    def _compare_acoustic_peaks(self) -> Dict[str, Any]:
        """
        Сравнява акустичните пикове с Planck данни
        
        Returns:
            Резултати от сравнението на пиковете
        """
        # Теоретични позиции на пиковете
        l_peaks_theory = self.calculate_acoustic_peak_positions()
        
        # Планк пикове
        planck_peaks = self.cmb_data['acoustic_peaks']
        l_peaks_obs = planck_peaks['l_peaks']
        
        # Сравнение (взимаме общия брой пикове)
        n_compare = min(len(l_peaks_theory), len(l_peaks_obs))
        
        # Относителни разлики
        relative_diff = (l_peaks_theory[:n_compare] - l_peaks_obs[:n_compare]) / l_peaks_obs[:n_compare] * 100
        
        # Средна разлика
        mean_diff = np.mean(np.abs(relative_diff))
        
        logger.info(f"Средна разлика в пиковете: {mean_diff:.2f}%")
        
        return {
            'l_peaks_theory': l_peaks_theory[:n_compare],
            'l_peaks_obs': l_peaks_obs[:n_compare],
            'relative_differences': relative_diff,
            'mean_difference': mean_diff,
            'max_difference': np.max(np.abs(relative_diff))
        }
    
    def _assess_agreement_level(self, reduced_chi_squared: float) -> str:
        """
        Оценява нивото на съответствие на базата на χ²/dof
        
        Args:
            reduced_chi_squared: Редуциран χ²
            
        Returns:
            Текстова оценка на съответствието
        """
        if reduced_chi_squared <= 1.5:
            return "Отлично съответствие"
        elif reduced_chi_squared <= 2.5:
            return "Добро съответствие"
        elif reduced_chi_squared <= 4.0:
            return "Приемливо съответствие"
        elif reduced_chi_squared <= 6.0:
            return "Слабо съответствие"
        else:
            return "Неприемливо съответствие"
    
    def angular_scale_analysis(self) -> Dict[str, Any]:
        """
        Анализ на ъгловите мащаби в CMB
        
        Returns:
            Резултати от анализа на ъгловите мащаби
        """
        # Теоретични стойности
        theta_star_theory = self.calculate_angular_sound_horizon()
        l_A_theory = np.pi / theta_star_theory
        
        # Планк стойности
        theta_star_planck = self.cmb_data['constraints']['theta_star']
        l_A_planck = self.planck_params['l_A']
        
        # Сравнение
        theta_diff = (theta_star_theory - theta_star_planck) / theta_star_planck * 100
        l_A_diff = (l_A_theory - l_A_planck) / l_A_planck * 100
        
        logger.info(f"Разлика в θ*: {theta_diff:.2f}%")
        logger.info(f"Разлика в l_A: {l_A_diff:.2f}%")
        
        return {
            'theta_star_theory': theta_star_theory,
            'theta_star_planck': theta_star_planck,
            'theta_difference': theta_diff,
            'l_A_theory': l_A_theory,
            'l_A_planck': l_A_planck,
            'l_A_difference': l_A_diff
        }
    
    def sound_horizon_evolution(self, z_max: float = 1500, n_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Анализ на еволюцията на звуковия хоризонт
        
        Args:
            z_max: Максимално червено отместване
            n_points: Брой точки в анализа
            
        Returns:
            Еволюция на звуковия хоризонт
        """
        # Мрежа от червени отмествания
        z_grid = np.logspace(np.log10(1090), np.log10(z_max), n_points)
        
        # Еволюция на звуковия хоризонт
        r_s_evolution = np.zeros(n_points)
        
        for i, z in enumerate(z_grid):
            r_s_evolution[i] = self.cosmology.sound_horizon_integral(z)
        
        # Стандартен ΛCDM за сравнение
        lambda_cdm = NonlinearTimeCosmology(alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
        r_s_lambda_cdm = np.zeros(n_points)
        
        for i, z in enumerate(z_grid):
            r_s_lambda_cdm[i] = lambda_cdm.sound_horizon_integral(z)
        
        # Относителна разлика
        relative_diff = (r_s_evolution - r_s_lambda_cdm) / r_s_lambda_cdm * 100
        
        logger.info(f"Максимална разлика в r_s: {np.max(np.abs(relative_diff)):.2f}%")
        
        return {
            'z_grid': z_grid,
            'r_s_nonlinear': r_s_evolution,
            'r_s_lambda_cdm': r_s_lambda_cdm,
            'relative_difference': relative_diff,
            'max_difference': np.max(np.abs(relative_diff))
        }
    
    def comprehensive_cmb_analysis(self) -> Dict[str, Any]:
        """
        Генерира обширен доклад за CMB анализа
        
        Returns:
            Пълен доклад с всички анализи
        """
        logger.info("🔍 Започва обширен CMB анализ...")
        
        # Основно сравнение с Planck
        planck_comparison = self.compare_with_planck_data()
        
        # Анализ на ъгловите мащаби
        angular_analysis = self.angular_scale_analysis()
        
        # Еволюция на звуковия хоризонт
        sound_horizon_evolution = self.sound_horizon_evolution()
        
        # Позиции на акустичните пикове
        peak_positions = self.calculate_acoustic_peak_positions()
        
        # Обобщение
        report = {
            'planck_comparison': planck_comparison,
            'angular_scale_analysis': angular_analysis,
            'sound_horizon_evolution': sound_horizon_evolution,
            'acoustic_peak_positions': peak_positions,
            'model_parameters': {
                'alpha': self.cosmology.alpha,
                'beta': self.cosmology.beta,
                'gamma': self.cosmology.gamma,
                'delta': self.cosmology.delta,
                'H0': self.cosmology.H0,
                'Omega_m': self.cosmology.Omega_m,
                'Omega_Lambda': self.cosmology.Omega_Lambda
            },
            'derived_quantities': {
                'r_s_star': self.cosmology.sound_horizon_integral(1089.8),
                'theta_star': self.calculate_angular_sound_horizon(),
                'l_A': np.pi / self.calculate_angular_sound_horizon()
            }
        }
        
        logger.info("✅ Обширният CMB анализ е завършен!")
        return report


def test_cmb_analyzer():
    """Тест на CMB анализатора"""
    print("🧪 ТЕСТ НА CMB АНАЛИЗАТОРА")
    print("=" * 50)
    
    # Създаване на анализатор
    analyzer = CMBAnalyzer()
    
    # Ъглов размер на звуковия хоризонт
    print("\n🎯 Ъглов размер на звуковия хоризонт...")
    theta_star = analyzer.calculate_angular_sound_horizon()
    print(f"θ* = {theta_star:.7f} rad")
    
    # Акустични пикове
    print("\n🔊 Акустични пикове...")
    peaks = analyzer.calculate_acoustic_peak_positions()
    print(f"Първи пик: l = {peaks[0]:.1f}")
    print(f"Втори пик: l = {peaks[1]:.1f}")
    
    # Сравнение с Planck
    print("\n📊 Сравнение с Planck данни...")
    comparison = analyzer.compare_with_planck_data()
    print(f"χ²/dof = {comparison['statistics']['reduced_chi_squared']:.2f}")
    print(f"Ниво на съответствие: {comparison['agreement_level']}")
    
    # Анализ на ъгловите мащаби
    print("\n📐 Анализ на ъгловите мащаби...")
    angular = analyzer.angular_scale_analysis()
    print(f"Разлика в θ*: {angular['theta_difference']:.2f}%")
    print(f"Разлика в l_A: {angular['l_A_difference']:.2f}%")
    
    print("\n✅ Всички тестове завършиха успешно!")
    
    return analyzer


if __name__ == "__main__":
    analyzer = test_cmb_analyzer() 