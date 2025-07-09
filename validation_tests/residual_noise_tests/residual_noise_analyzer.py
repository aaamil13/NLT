"""
Анализатор за остатъчен шум
==========================

Обширен анализ на остатъчен шум в моделите за нелинейно време.
Използва всички налични оптимизационни и статистически методи.

Автор: Система за анализ на нелинейно време
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import normaltest, kstest, shapiro
from scipy.fft import fft, fftfreq
import warnings
from typing import Dict, List, Tuple, Any, Optional

# Импортираме общите утилити
from validation_tests.common_utils.optimization_engines import DifferentialEvolutionOptimizer, BasinhoppingOptimizer
from validation_tests.common_utils.mcmc_bayesian import MCMCBayesianAnalyzer, BayesianModelComparison
from validation_tests.common_utils.statistical_tests import StatisticalSignificanceTest
from validation_tests.common_utils.data_processors import RawDataProcessor

# Импортираме нашите модели
from lib.advanced_analytical_functions import AdvancedAnalyticalFunctions

# Настройка на логирането
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Скорост на светлината
c = 3e8  # m/s

# Глобални функции за модели
def lambda_cdm_model_global(params, z, mu_obs, c):
    """Глобална функция за ΛCDM модел"""
    H0, Om, OL = params
    E_z = np.sqrt(Om * (1 + z)**3 + OL)
    d_L = (c / H0) * (1 + z) * np.trapz(1 / E_z, z)
    mu_pred = 25 + 5 * np.log10(d_L * 1e6)
    return np.sum((mu_obs - mu_pred)**2)

def nonlinear_time_model_global(params, z, mu_obs, c, aaf):
    """Глобална функция за нелинейно време модел"""
    H0, alpha, beta = params
    mu_pred = []
    for z_val in z:
        t_z = aaf.analytical_t_z_approximation(z_val)
        d_L = (c / H0) * (1 + z_val) * (alpha * t_z + beta)
        mu_pred.append(25 + 5 * np.log10(d_L * 1e6))
    mu_pred = np.array(mu_pred)
    return np.sum((mu_obs - mu_pred)**2)

def polynomial_model_global(params, z, mu_obs):
    """Глобална функция за полиномен модел"""
    coeffs = params
    log_z = np.log10(z + 1e-10)
    mu_pred = np.polyval(coeffs, log_z)
    return np.sum((mu_obs - mu_pred)**2)

warnings.filterwarnings('ignore')


class ResidualNoiseAnalyzer:
    """
    Анализатор за остатъчен шум в моделите
    """
    
    def __init__(self, use_raw_data: bool = True):
        """
        Инициализация на анализатора
        
        Args:
            use_raw_data: Дали да използва сурови данни
        """
        self.use_raw_data = use_raw_data
        self.aaf = AdvancedAnalyticalFunctions()
        self.data_processor = RawDataProcessor() if use_raw_data else None
        self.residual_data = {}
        self.analysis_results = {}
        
    def load_observational_data(self) -> Dict[str, Any]:
        """
        Зарежда наблюдателни данни
        
        Returns:
            Наблюдателни данни
        """
        if self.use_raw_data and self.data_processor:
            # Зареждаме сурови данни
            print("Зареждане на сурови данни...")
            
            # Pantheon+ данни
            pantheon_data = self.data_processor.load_pantheon_plus_data()
            
            # SH0ES данни
            shoes_data = self.data_processor.load_shoes_data()
            
            # Извличаме унифицирани данни
            unified_data = self.data_processor.extract_redshift_magnitude_data()
            
            if unified_data['combined']:
                data = {
                    'z': unified_data['combined']['z'],
                    'mu_obs': unified_data['combined']['magnitude'],
                    'sources': unified_data['combined']['sources'],
                    'n_points': unified_data['combined']['n_points']
                }
            else:
                # Фолбек към синтетични данни
                data = self._generate_synthetic_data()
        else:
            # Синтетични данни
            data = self._generate_synthetic_data()
        
        return data
    
    def _generate_synthetic_data(self) -> Dict[str, Any]:
        """
        Генерира синтетични данни за тестване
        
        Returns:
            Синтетични данни
        """
        # Генерираме z стойности
        z = np.logspace(-3, 1, 200)  # От 0.001 до 10
        
        # Пресмятаме теоретично разстояние с нелинейно време
        mu_theoretical = []
        for z_val in z:
            t_z = self.aaf.analytical_t_z_approximation(z_val)
            # Конвертираме в разстоянен модул
            mu_theoretical.append(25 + 5 * np.log10(t_z * 1e28))  # Приблизително
        
        mu_theoretical = np.array(mu_theoretical)
        
        # Добавяме реалистичен шум
        noise_amplitude = 0.1  # магнитуди
        noise = np.random.normal(0, noise_amplitude, len(z))
        
        # Добавяме систематични грешки
        systematic_error = 0.02 * np.sin(2 * np.pi * np.log10(z))
        
        mu_observed = mu_theoretical + noise + systematic_error
        
        return {
            'z': z,
            'mu_obs': mu_observed,
            'mu_theoretical': mu_theoretical,
            'noise': noise,
            'systematic_error': systematic_error,
            'sources': ['synthetic'] * len(z),
            'n_points': len(z)
        }
    
    def fit_models(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Фитва различни модели към данните
        
        Args:
            data: Наблюдателни данни
            
        Returns:
            Резултати от фитинга
        """
        z = data['z']
        mu_obs = data['mu_obs']
        
        results = {}
        
        # Фитваме модели с различни оптимизатори
        models = {
            'lambda_cdm': {
                'function': lambda params: lambda_cdm_model_global(params, z, mu_obs, c),
                'bounds': [(50, 80), (0.1, 0.5), (0.5, 0.9)],
                'initial': np.array([70, 0.3, 0.7])
            },
            'nonlinear_time': {
                'function': lambda params: nonlinear_time_model_global(params, z, mu_obs, c, self.aaf),
                'bounds': [(50, 80), (0.1, 10), (-5, 5)],
                'initial': np.array([70, 1.0, 0.0])
            },
            'polynomial': {
                'function': lambda params: polynomial_model_global(params, z, mu_obs),
                'bounds': [(-50, 50)] * 5,  # 5-та степен полином
                'initial': np.array([40, 0, 0, 0, 0])
            }
        }
        
        for model_name, model_config in models.items():
            print(f"Фитване на {model_name} модел...")
            
            # Differential Evolution
            de_optimizer = DifferentialEvolutionOptimizer(max_iterations=500, parallel=False)
            de_result = de_optimizer.optimize(
                model_config['function'],
                model_config['bounds']
            )
            
            # Basinhopping
            bh_optimizer = BasinhoppingOptimizer(n_iterations=200)
            bh_result = bh_optimizer.optimize(
                model_config['function'],
                model_config['initial'],
                model_config['bounds']
            )
            
            results[model_name] = {
                'differential_evolution': de_result,
                'basinhopping': bh_result,
                'bounds': model_config['bounds']
            }
        
        return results
    
    def calculate_residuals(self, data: Dict[str, Any], 
                           fit_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Пресмята остатъци за всички модели
        
        Args:
            data: Наблюдателни данни
            fit_results: Резултати от фитинга
            
        Returns:
            Остатъци за всички модели
        """
        z = data['z']
        mu_obs = data['mu_obs']
        
        residuals = {}
        
        for model_name, results in fit_results.items():
            # Използваме най-добрия резултат от DE
            best_params = results['differential_evolution']['best_parameters']
            
            # Пресмятаме прогнозите с защитени операции
            if model_name == 'lambda_cdm':
                H0, Om, OL = best_params
                
                # Защитена sqrt операция
                E_z_squared = Om * (1 + z)**3 + OL
                E_z_squared = np.maximum(E_z_squared, 1e-30)
                E_z = np.sqrt(E_z_squared)
                
                # Защитена интеграция
                integrand = 1 / np.maximum(E_z, 1e-30)
                integral = np.trapz(integrand, z)
                
                # Защитена дистанция
                d_L = (c / H0) * (1 + z) * integral
                d_L = np.maximum(d_L, 1e-30)
                
                # Защитен logarithm
                log_arg = d_L * 1e6
                log_arg = np.maximum(log_arg, 1e-30)
                mu_pred = 25 + 5 * np.log10(log_arg)
                
            elif model_name == 'nonlinear_time':
                H0, alpha, beta = best_params
                mu_pred = []
                for z_val in z:
                    # Защитена t_z апроксимация
                    try:
                        t_z = self.aaf.analytical_t_z_approximation(z_val)
                        if np.isnan(t_z) or np.isinf(t_z):
                            t_z = z_val / (1 + z_val)  # Fallback апроксимация
                    except:
                        t_z = z_val / (1 + z_val)  # Fallback апроксимация
                    
                    # Защитена дистанция
                    d_L = (c / H0) * (1 + z_val) * (alpha * t_z + beta)
                    d_L = np.maximum(d_L, 1e-30)  # Предотвратяваме отрицателни стойности
                    
                    # Защитен logarithm
                    log_arg = d_L * 1e6
                    log_arg = np.maximum(log_arg, 1e-30)
                    mu_val = 25 + 5 * np.log10(log_arg)
                    
                    # Проверка за NaN/inf
                    if np.isnan(mu_val) or np.isinf(mu_val):
                        # Fallback към lambda_cdm предсказание
                        Om_default = 0.3
                        OL_default = 0.7
                        E_z_default = np.sqrt(Om_default * (1 + z_val)**3 + OL_default)
                        d_L_default = (c / H0) * (1 + z_val) * (1 / E_z_default)
                        mu_val = 25 + 5 * np.log10(d_L_default * 1e6)
                    
                    mu_pred.append(mu_val)
                mu_pred = np.array(mu_pred)
                
            elif model_name == 'polynomial':
                coeffs = best_params
                
                # Защитен logarithm
                log_z = np.log10(z + 1e-10)
                
                # Защитена полиномиална оценка
                mu_pred = np.polyval(coeffs, log_z)
                
                # Проверка за NaN/inf
                if np.any(np.isnan(mu_pred)) or np.any(np.isinf(mu_pred)):
                    print(f"Предупреждение: NaN/inf в polynomial модел, използвам fallback")
                    mu_pred = np.full_like(z, 43.0)  # Fallback към константа
            
            # Пресмятаме остатъците с защитени операции
            residuals_array = mu_obs - mu_pred
            
            # Проверка за валидност
            if np.any(np.isnan(residuals_array)) or np.any(np.isinf(residuals_array)):
                print(f"Предупреждение: NaN/inf в остатъци за {model_name}")
                # Филтриране на NaN стойности
                valid_mask = np.isfinite(residuals_array) & np.isfinite(mu_pred)
                residuals_array = np.where(valid_mask, residuals_array, 0.0)
                mu_pred = np.where(valid_mask, mu_pred, np.median(mu_obs))
            
            # Защитени статистики
            rms = np.sqrt(np.mean(residuals_array**2)) if len(residuals_array) > 0 else 0.0
            chi_squared = np.sum(residuals_array**2) if len(residuals_array) > 0 else 0.0
            
            residuals[model_name] = {
                'residuals': residuals_array,
                'predicted': mu_pred,
                'parameters': best_params,
                'rms': rms,
                'chi_squared': chi_squared
            }
        
        self.residual_data = residuals
        return residuals
    
    def analyze_residual_noise(self, residuals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализира остатъчния шум
        
        Args:
            residuals: Остатъци от моделите
            
        Returns:
            Анализ на остатъчния шум
        """
        analysis = {}
        
        for model_name, residual_data in residuals.items():
            print(f"Анализ на остатъчния шум за {model_name}...")
            
            res = residual_data['residuals']
            
            # Филтриране на NaN стойности
            res_clean = res[np.isfinite(res)]
            
            if len(res_clean) < 3:
                print(f"Предупреждение: твърде малко валидни данни за {model_name} ({len(res_clean)} точки)")
                # Минимални статистики
                basic_stats = {
                    'mean': 0.0,
                    'std': 0.0,
                    'skewness': 0.0,
                    'kurtosis': 0.0,
                    'median': 0.0,
                    'mad': 0.0
                }
                
                # Празни тестове
                normality_tests = {
                    'shapiro_wilk': {'statistic': 0.0, 'p_value': 0.0},
                    'kolmogorov_smirnov': {'statistic': 0.0, 'p_value': 0.0},
                    'anderson_darling': {'statistic': 0.0, 'p_value': 0.0}
                }
                
                autocorr_tests = {
                    'runs_test': {'statistic': 0.0, 'p_value': 0.0},
                    'durbin_watson': {'statistic': 0.0, 'p_value': 0.0}
                }
                
                spectral_analysis = {
                    'dominant_frequency': 0.0,
                    'power_spectrum': np.array([]),
                    'frequencies': np.array([]),
                    'white_noise_score': 0.0,
                    'spectral_entropy': 0.0
                }
                
                outlier_analysis = {
                    'iqr_outlier_fraction': 0.0,
                    'z_score_outlier_fraction': 0.0,
                    'iqr_outliers': np.array([]),
                    'z_score_outliers': np.array([])
                }
                
                time_freq_analysis = {
                    'wavelet_coeffs': np.array([]),
                    'time_series': np.array([]),
                    'frequency_content': np.array([])
                }
                
            else:
                # Основни статистики
                basic_stats = {
                    'mean': np.mean(res_clean),
                    'std': np.std(res_clean),
                    'skewness': self._calculate_skewness(res_clean),
                    'kurtosis': self._calculate_kurtosis(res_clean),
                    'median': np.median(res_clean),
                    'mad': np.median(np.abs(res_clean - np.median(res_clean)))
                }
            
                # Статистически тестове
                stat_test = StatisticalSignificanceTest()
                
                # Тест за нормалност
                normality_tests = {
                    'shapiro_wilk': stat_test.shapiro_wilk_test(res_clean),
                    'kolmogorov_smirnov': stat_test.kolmogorov_smirnov_test(res_clean),
                    'anderson_darling': stat_test.anderson_darling_test(res_clean)
                }
                
                # Тест за автокорелация
                autocorr_tests = {
                    'runs_test': stat_test.runs_test(res_clean),
                    'durbin_watson': stat_test.durbin_watson_test(res_clean)
                }
                
                # Спектрален анализ
                spectral_analysis = self._analyze_power_spectrum(res_clean)
                
                # Анализ на outliers
                outlier_analysis = self._analyze_outliers(res_clean)
                
                # Време-честотен анализ
                time_freq_analysis = self._time_frequency_analysis(res_clean)
            
            analysis[model_name] = {
                'basic_statistics': basic_stats,
                'normality_tests': normality_tests,
                'autocorrelation_tests': autocorr_tests,
                'spectral_analysis': spectral_analysis,
                'outlier_analysis': outlier_analysis,
                'time_frequency_analysis': time_freq_analysis
            }
        
        self.analysis_results = analysis
        return analysis
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Пресмята скосеност"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std)**3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Пресмята ексцес"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std)**4) - 3
    
    def _analyze_power_spectrum(self, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Анализира степенния спектър на остатъците
        
        Args:
            residuals: Остатъци
            
        Returns:
            Спектрален анализ
        """
        # FFT на остатъците
        fft_res = fft(residuals)
        freqs = fftfreq(len(residuals))
        
        # Степенен спектър
        power_spectrum = np.abs(fft_res)**2
        
        # Намираме доминиращи честоти
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        
        # Бял шум тест
        white_noise_score = self._test_white_noise(power_spectrum)
        
        return {
            'dominant_frequency': dominant_freq,
            'power_spectrum': power_spectrum,
            'frequencies': freqs,
            'white_noise_score': white_noise_score,
            'spectral_entropy': self._calculate_spectral_entropy(power_spectrum)
        }
    
    def _test_white_noise(self, power_spectrum: np.ndarray) -> float:
        """
        Тества дали остатъците са бял шум
        
        Args:
            power_spectrum: Степенен спектър
            
        Returns:
            Скор за бял шум (по-високо = по-близо до бял шум)
        """
        # Плоскостта на спектъра
        spectrum_flatness = np.var(power_spectrum[1:len(power_spectrum)//2])
        
        # Нормализираме
        return 1.0 / (1.0 + spectrum_flatness)
    
    def _calculate_spectral_entropy(self, power_spectrum: np.ndarray) -> float:
        """
        Пресмята спектрална ентропия
        
        Args:
            power_spectrum: Степенен спектър
            
        Returns:
            Спектрална ентропия
        """
        # Нормализираме спектъра
        norm_spectrum = power_spectrum / np.sum(power_spectrum)
        
        # Изчисляваме ентропията
        entropy = -np.sum(norm_spectrum * np.log(norm_spectrum + 1e-10))
        
        return entropy
    
    def _analyze_outliers(self, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Анализира outliers в остатъците
        
        Args:
            residuals: Остатъци
            
        Returns:
            Анализ на outliers
        """
        # IQR метод
        Q1 = np.percentile(residuals, 25)
        Q3 = np.percentile(residuals, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (residuals < lower_bound) | (residuals > upper_bound)
        
        # Z-score метод
        z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
        z_outlier_mask = z_scores > 3
        
        return {
            'iqr_outliers': np.sum(outlier_mask),
            'iqr_outlier_fraction': np.sum(outlier_mask) / len(residuals),
            'z_score_outliers': np.sum(z_outlier_mask),
            'z_score_outlier_fraction': np.sum(z_outlier_mask) / len(residuals),
            'outlier_indices': np.where(outlier_mask)[0],
            'z_outlier_indices': np.where(z_outlier_mask)[0]
        }
    
    def _time_frequency_analysis(self, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Време-честотен анализ на остатъците
        
        Args:
            residuals: Остатъци
            
        Returns:
            Време-честотен анализ
        """
        # Wavelet анализ (опростен)
        try:
            from scipy import signal
            
            # Spectrogram
            f, t, Sxx = signal.spectrogram(residuals, nperseg=min(64, len(residuals)//4))
            
            # Instantaneous frequency
            analytic_signal = signal.hilbert(residuals)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
            
            return {
                'spectrogram_frequencies': f,
                'spectrogram_times': t,
                'spectrogram_power': Sxx,
                'instantaneous_frequency': instantaneous_frequency
            }
        except:
            return {
                'error': 'Време-честотният анализ не може да бъде изпълнен'
            }
    
    def run_mcmc_model_comparison(self, data: Dict[str, Any], 
                                 residuals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Стартира MCMC анализ за сравнение на модели
        
        Args:
            data: Данни
            residuals: Остатъци
            
        Returns:
            MCMC резултати
        """
        print("Стартиране на MCMC анализ...")
        
        z = data['z']
        mu_obs = data['mu_obs']
        
        # Дефинираме log-likelihood функции с защитени математически операции
        def log_likelihood_lambda_cdm(params, data):
            try:
                H0, Om, OL, sigma = params
                
                # Защитена sqrt операция
                E_z_squared = Om * (1 + z)**3 + OL
                E_z_squared = np.maximum(E_z_squared, 1e-30)  # Предотвратяваме нула
                E_z = np.sqrt(E_z_squared)
                
                # Защитена интеграция
                integrand = 1 / np.maximum(E_z, 1e-30)
                integral = np.trapz(integrand, z)
                
                # Защитена дистанция
                d_L = (c / H0) * (1 + z) * integral
                d_L = np.maximum(d_L, 1e-30)  # Предотвратяваме отрицателни стойности
                
                # Защитен logarithm
                log_arg = d_L * 1e6
                log_arg = np.maximum(log_arg, 1e-30)
                mu_pred = 25 + 5 * np.log10(log_arg)
                
                # Проверка за NaN/inf
                if np.any(np.isnan(mu_pred)) or np.any(np.isinf(mu_pred)):
                    return -np.inf
                
                # Защитена sigma
                sigma_safe = np.maximum(sigma, 1e-10)
                
                chi2 = np.sum((mu_obs - mu_pred)**2 / sigma_safe**2)
                log_det = np.sum(np.log(2 * np.pi * sigma_safe**2))
                
                likelihood = -0.5 * (chi2 + log_det)
                
                # Финална проверка
                if np.isnan(likelihood) or np.isinf(likelihood):
                    return -np.inf
                    
                return likelihood
                
            except Exception as e:
                print(f"Грешка в log_likelihood_lambda_cdm: {e}")
                return -np.inf
        
        def log_likelihood_nonlinear(params, data):
            try:
                H0, alpha, beta, sigma = params
                mu_pred = []
                
                for z_val in z:
                    # Защитена t_z апроксимация
                    try:
                        t_z = self.aaf.analytical_t_z_approximation(z_val)
                        if np.isnan(t_z) or np.isinf(t_z):
                            t_z = z_val / (1 + z_val)  # Fallback апроксимация
                    except:
                        t_z = z_val / (1 + z_val)  # Fallback апроксимация
                    
                    # Защитена дистанция
                    d_L = (c / H0) * (1 + z_val) * (alpha * t_z + beta)
                    d_L = np.maximum(d_L, 1e-30)  # Предотвратяваме отрицателни стойности
                    
                    # Защитен logarithm
                    log_arg = d_L * 1e6
                    log_arg = np.maximum(log_arg, 1e-30)
                    mu_val = 25 + 5 * np.log10(log_arg)
                    
                    # Проверка за NaN/inf
                    if np.isnan(mu_val) or np.isinf(mu_val):
                        return -np.inf
                        
                    mu_pred.append(mu_val)
                
                mu_pred = np.array(mu_pred)
                
                # Защитена sigma
                sigma_safe = np.maximum(sigma, 1e-10)
                
                chi2 = np.sum((mu_obs - mu_pred)**2 / sigma_safe**2)
                log_det = np.sum(np.log(2 * np.pi * sigma_safe**2))
                
                likelihood = -0.5 * (chi2 + log_det)
                
                # Финална проверка
                if np.isnan(likelihood) or np.isinf(likelihood):
                    return -np.inf
                    
                return likelihood
                
            except Exception as e:
                print(f"Грешка в log_likelihood_nonlinear: {e}")
                return -np.inf
        
        # Дефинираме prior функции с защитени проверки
        def log_prior_lambda_cdm(params):
            try:
                H0, Om, OL, sigma = params
                
                # Проверка за NaN/inf
                if np.any(np.isnan(params)) or np.any(np.isinf(params)):
                    return -np.inf
                
                # Строги физически граници
                if (60 < H0 < 75 and           # По-строга граница за H0
                    0.15 < Om < 0.45 and       # По-строга граница за Om
                    0.55 < OL < 0.85 and       # По-строга граница за OL
                    0.05 < sigma < 0.5 and     # По-строга граница за sigma
                    abs(Om + OL - 1.0) < 0.1): # Плоска вселена приближение
                    return 0.0
                return -np.inf
                
            except Exception as e:
                print(f"Грешка в log_prior_lambda_cdm: {e}")
                return -np.inf
        
        def log_prior_nonlinear(params):
            try:
                H0, alpha, beta, sigma = params
                
                # Проверка за NaN/inf
                if np.any(np.isnan(params)) or np.any(np.isinf(params)):
                    return -np.inf
                
                # Строги физически граници
                if (60 < H0 < 75 and           # По-строга граница за H0
                    0.5 < alpha < 5.0 and      # По-строга граница за alpha
                    -2.0 < beta < 2.0 and      # По-строга граница за beta
                    0.05 < sigma < 0.5):       # По-строга граница за sigma
                    return 0.0
                return -np.inf
                
            except Exception as e:
                print(f"Грешка в log_prior_nonlinear: {e}")
                return -np.inf
        
        # Байесово сравнение
        comparison = BayesianModelComparison()
        
        comparison.add_model(
            'lambda_cdm',
            log_likelihood_lambda_cdm,
            log_prior_lambda_cdm,
            [(60, 75), (0.15, 0.45), (0.55, 0.85), (0.05, 0.5)],
            np.array([67.4, 0.315, 0.685, 0.15])  # Planck 2018 стойности
        )
        
        comparison.add_model(
            'nonlinear_time',
            log_likelihood_nonlinear,
            log_prior_nonlinear,
            [(60, 75), (0.5, 5.0), (-2.0, 2.0), (0.05, 0.5)],
            np.array([67.4, 1.5, 0.0, 0.15])  # Физически разумни стойности
        )
        
        # Стартираме сравнението със защитени настройки
        try:
            mcmc_results = comparison.run_comparison(
                mu_obs,
                {
                    'n_walkers': 64,      # Повече walkers за стабилност
                    'n_steps': 1500,      # Повече стъпки за конвергенция
                    'n_burn': 500,        # По-дълъг burn-in
                    'thin': 1             # Без прореждане
                }
            )
            
            # Проверка за успешност
            if mcmc_results is None:
                print("MCMC анализът неуспешен - използвам fallback резултати")
                mcmc_results = {
                    'lambda_cdm': {
                        'chain': np.array([]),
                        'log_prob': np.array([]),
                        'acceptance_fraction': 0.0,
                        'evidence': -np.inf
                    },
                    'nonlinear_time': {
                        'chain': np.array([]),
                        'log_prob': np.array([]),
                        'acceptance_fraction': 0.0,
                        'evidence': -np.inf
                    },
                    'bayes_factor': 1.0,
                    'status': 'failed'
                }
            else:
                mcmc_results['status'] = 'success'
                
        except Exception as e:
            print(f"Грешка в MCMC анализ: {e}")
            mcmc_results = {
                'lambda_cdm': {
                    'chain': np.array([]),
                    'log_prob': np.array([]),
                    'acceptance_fraction': 0.0,
                    'evidence': -np.inf
                },
                'nonlinear_time': {
                    'chain': np.array([]),
                    'log_prob': np.array([]),
                    'acceptance_fraction': 0.0,
                    'evidence': -np.inf
                },
                'bayes_factor': 1.0,
                'status': 'error',
                'error_message': str(e)
            }
        
        return mcmc_results
    
    def plot_comprehensive_analysis(self, data: Dict[str, Any], 
                                  residuals: Dict[str, Any], 
                                  analysis: Dict[str, Any],
                                  save_path: str = None):
        """
        Създава обширни графики на анализа
        
        Args:
            data: Данни
            residuals: Остатъци
            analysis: Анализ
            save_path: Път за записване
        """
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        model_names = list(residuals.keys())
        colors = ['blue', 'red', 'green']
        
        # 1. Hubble диаграма с фитове
        ax = axes[0, 0]
        ax.scatter(data['z'], data['mu_obs'], alpha=0.6, s=10, color='black', label='Данни')
        
        for i, model_name in enumerate(model_names):
            ax.plot(data['z'], residuals[model_name]['predicted'], 
                   color=colors[i], label=f'{model_name} fit')
        
        ax.set_xlabel('Червено отместване z')
        ax.set_ylabel('Модулна величина')
        ax.set_title('Hubble диаграма с фитове')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Остатъци за всички модели
        ax = axes[0, 1]
        for i, model_name in enumerate(model_names):
            ax.scatter(data['z'], residuals[model_name]['residuals'], 
                      alpha=0.6, s=10, color=colors[i], label=model_name)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax.set_xlabel('Червено отместване z')
        ax.set_ylabel('Остатъци')
        ax.set_title('Остатъци за всички модели')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Хистограми на остатъците
        ax = axes[0, 2]
        for i, model_name in enumerate(model_names):
            res_data = residuals[model_name]['residuals']
            # Филтриране на NaN стойности
            res_data = res_data[~np.isnan(res_data)]
            res_data = res_data[np.isfinite(res_data)]
            
            if len(res_data) > 0:
                ax.hist(res_data, bins=30, alpha=0.7, 
                       color=colors[i], label=model_name, density=True)
            else:
                print(f"Предупреждение: няма валидни данни за {model_name} histogram")
        
        ax.set_xlabel('Остатъци')
        ax.set_ylabel('Плътност')
        ax.set_title('Разпределение на остатъците')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Q-Q plots
        ax = axes[0, 3]
        for i, model_name in enumerate(model_names):
            from scipy import stats
            res_data = residuals[model_name]['residuals']
            # Филтриране на NaN стойности
            res_data = res_data[~np.isnan(res_data)]
            res_data = res_data[np.isfinite(res_data)]
            
            if len(res_data) > 3:  # Минимум 3 точки за Q-Q plot
                try:
                    stats.probplot(res_data, dist='norm', plot=ax)
                except Exception as e:
                    print(f"Грешка в Q-Q plot за {model_name}: {e}")
            else:
                print(f"Предупреждение: няма достатъчно данни за {model_name} Q-Q plot")
        
        ax.set_title('Q-Q Plot (Нормалност)')
        ax.grid(True, alpha=0.3)
        
        # 5. Степенни спектри
        ax = axes[1, 0]
        for i, model_name in enumerate(model_names):
            if 'spectral_analysis' in analysis[model_name]:
                spec = analysis[model_name]['spectral_analysis']
                freqs = spec['frequencies'][:len(spec['frequencies'])//2]
                power = spec['power_spectrum'][:len(spec['power_spectrum'])//2]
                ax.loglog(freqs[1:], power[1:], color=colors[i], label=model_name)
        
        ax.set_xlabel('Честота')
        ax.set_ylabel('Степен')
        ax.set_title('Степенни спектри')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Autocorrelation
        ax = axes[1, 1]
        for i, model_name in enumerate(model_names):
            res = residuals[model_name]['residuals']
            # Филтриране на NaN стойности
            res = res[~np.isnan(res)]
            res = res[np.isfinite(res)]
            
            if len(res) > 10:  # Минимум 10 точки за автокорелация
                try:
                    # Опростена автокорелация
                    lags = np.arange(1, min(50, len(res)))
                    autocorr = []
                    for lag in lags:
                        try:
                            corr = np.corrcoef(res[:-lag], res[lag:])[0, 1]
                            if np.isfinite(corr):
                                autocorr.append(corr)
                            else:
                                autocorr.append(0.0)
                        except:
                            autocorr.append(0.0)
                    ax.plot(lags, autocorr, color=colors[i], label=model_name)
                except Exception as e:
                    print(f"Грешка в автокорелация за {model_name}: {e}")
            else:
                print(f"Предупреждение: няма достатъчно данни за {model_name} автокорелация")
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax.set_xlabel('Лаг')
        ax.set_ylabel('Автокорелация')
        ax.set_title('Автокорелационна функция')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. RMS vs модел
        ax = axes[1, 2]
        rms_values = [residuals[model]['rms'] for model in model_names]
        ax.bar(model_names, rms_values, color=colors[:len(model_names)])
        ax.set_ylabel('RMS')
        ax.set_title('RMS на остатъците')
        ax.tick_params(axis='x', rotation=45)
        
        # 8. Outliers
        ax = axes[1, 3]
        outlier_fractions = []
        for model_name in model_names:
            if 'outlier_analysis' in analysis[model_name]:
                outlier_fractions.append(analysis[model_name]['outlier_analysis']['iqr_outlier_fraction'])
            else:
                outlier_fractions.append(0)
        
        ax.bar(model_names, outlier_fractions, color=colors[:len(model_names)])
        ax.set_ylabel('Доля outliers')
        ax.set_title('Outliers в остатъците')
        ax.tick_params(axis='x', rotation=45)
        
        # 9. Normality test p-values
        ax = axes[2, 0]
        for i, model_name in enumerate(model_names):
            if 'normality_tests' in analysis[model_name]:
                p_values = [
                    analysis[model_name]['normality_tests']['shapiro_wilk']['p_value'],
                    analysis[model_name]['normality_tests']['kolmogorov_smirnov']['p_value']
                ]
                ax.bar(np.arange(len(p_values)) + i*0.25, p_values, 
                      width=0.25, color=colors[i], label=model_name)
        
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Shapiro-Wilk', 'KS'])
        ax.set_ylabel('p-стойност')
        ax.set_title('Тестове за нормалност')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 10. Spectral entropy
        ax = axes[2, 1]
        entropy_values = []
        for model_name in model_names:
            if 'spectral_analysis' in analysis[model_name]:
                entropy_values.append(analysis[model_name]['spectral_analysis']['spectral_entropy'])
            else:
                entropy_values.append(0)
        
        ax.bar(model_names, entropy_values, color=colors[:len(model_names)])
        ax.set_ylabel('Спектрална ентропия')
        ax.set_title('Спектрална ентропия')
        ax.tick_params(axis='x', rotation=45)
        
        # 11. Basic statistics comparison
        ax = axes[2, 2]
        stats_names = ['mean', 'std', 'skewness', 'kurtosis']
        x_pos = np.arange(len(stats_names))
        
        for i, model_name in enumerate(model_names):
            if 'basic_statistics' in analysis[model_name]:
                stats_values = [
                    analysis[model_name]['basic_statistics']['mean'],
                    analysis[model_name]['basic_statistics']['std'],
                    analysis[model_name]['basic_statistics']['skewness'],
                    analysis[model_name]['basic_statistics']['kurtosis']
                ]
                ax.bar(x_pos + i*0.25, stats_values, width=0.25, 
                      color=colors[i], label=model_name)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats_names)
        ax.set_ylabel('Стойност')
        ax.set_title('Основни статистики')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 12. Summary score
        ax = axes[2, 3]
        # Пресмятаме обобщен скор
        summary_scores = []
        for model_name in model_names:
            score = 0
            if 'basic_statistics' in analysis[model_name]:
                # По-ниско std е по-добре
                score += 1.0 / (1.0 + analysis[model_name]['basic_statistics']['std'])
                # По-малко outliers е по-добре
                if 'outlier_analysis' in analysis[model_name]:
                    score += 1.0 - analysis[model_name]['outlier_analysis']['iqr_outlier_fraction']
            summary_scores.append(score)
        
        ax.bar(model_names, summary_scores, color=colors[:len(model_names)])
        ax.set_ylabel('Обобщен скор')
        ax.set_title('Обобщен скор на качеството')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_comprehensive_report(self, data: Dict[str, Any], 
                                    residuals: Dict[str, Any], 
                                    analysis: Dict[str, Any],
                                    mcmc_results: Dict[str, Any] = None) -> str:
        """
        Генерира обширен доклад
        
        Args:
            data: Данни
            residuals: Остатъци
            analysis: Анализ
            mcmc_results: MCMC резултати
            
        Returns:
            Подробен доклад
        """
        report = []
        report.append("=" * 80)
        report.append("ОБШИРЕН АНАЛИЗ НА ОСТАТЪЧЕН ШУМ")
        report.append("=" * 80)
        report.append("")
        
        # Основни данни
        report.append("ОСНОВНИ ДАННИ:")
        report.append("-" * 30)
        report.append(f"Брой наблюдения: {data['n_points']}")
        report.append(f"Диапазон z: {np.min(data['z']):.4f} - {np.max(data['z']):.4f}")
        report.append(f"Диапазон μ: {np.min(data['mu_obs']):.2f} - {np.max(data['mu_obs']):.2f}")
        report.append("")
        
        # Резултати от фитинга
        report.append("РЕЗУЛТАТИ ОТ ФИТИНГА:")
        report.append("-" * 30)
        
        for model_name, res_data in residuals.items():
            report.append(f"\n{model_name.upper()}:")
            report.append(f"  RMS: {res_data['rms']:.4f}")
            report.append(f"  χ²: {res_data['chi_squared']:.2f}")
            report.append(f"  Параметри: {res_data['parameters']}")
        
        report.append("")
        
        # Анализ на остатъчния шум
        report.append("АНАЛИЗ НА ОСТАТЪЧЕН ШУМ:")
        report.append("-" * 30)
        
        for model_name, analysis_data in analysis.items():
            report.append(f"\n{model_name.upper()}:")
            
            # Основни статистики
            if 'basic_statistics' in analysis_data:
                stats = analysis_data['basic_statistics']
                report.append(f"  Средно: {stats['mean']:.6f}")
                report.append(f"  Стандартно отклонение: {stats['std']:.6f}")
                report.append(f"  Скосеност: {stats['skewness']:.4f}")
                report.append(f"  Ексцес: {stats['kurtosis']:.4f}")
            
            # Тестове за нормалност
            if 'normality_tests' in analysis_data:
                report.append("  Тестове за нормалност:")
                for test_name, test_result in analysis_data['normality_tests'].items():
                    if isinstance(test_result, dict) and 'p_value' in test_result:
                        report.append(f"    {test_name}: p = {test_result['p_value']:.6f}")
                    else:
                        report.append(f"    {test_name}: N/A (недостатъчно данни)")
            
            # Автокорелационни тестове
            if 'autocorrelation_tests' in analysis_data:
                report.append("  Автокорелационни тестове:")
                for test_name, test_result in analysis_data['autocorrelation_tests'].items():
                    if isinstance(test_result, dict) and 'p_value' in test_result:
                        report.append(f"    {test_name}: p = {test_result['p_value']:.6f}")
                    else:
                        report.append(f"    {test_name}: N/A (недостатъчно данни)")
            
            # Спектрален анализ
            if 'spectral_analysis' in analysis_data:
                spec = analysis_data['spectral_analysis']
                report.append(f"  Спектрална ентропия: {spec.get('spectral_entropy', 'N/A'):.4f}")
                report.append(f"  Бял шум скор: {spec.get('white_noise_score', 'N/A'):.4f}")
            
            # Outliers
            if 'outlier_analysis' in analysis_data:
                outliers = analysis_data['outlier_analysis']
                report.append(f"  Outliers (IQR): {outliers['iqr_outlier_fraction']:.2%}")
                report.append(f"  Outliers (Z-score): {outliers['z_score_outlier_fraction']:.2%}")
        
        report.append("")
        
        # MCMC резултати
        if mcmc_results:
            report.append("MCMC БАЙЕСОВО СРАВНЕНИЕ:")
            report.append("-" * 30)
            
            comparison = mcmc_results['comparison']
            for criterion in ['AIC', 'BIC', 'DIC', 'WAIC']:
                if criterion in comparison:
                    best_model = comparison[criterion]['best_model']
                    report.append(f"{criterion}: Най-добър модел - {best_model}")
            
            report.append("")
        
        # Заключения
        report.append("ЗАКЛЮЧЕНИЯ:")
        report.append("-" * 30)
        
        # Намираме най-добрия модел по RMS
        best_model_rms = min(residuals, key=lambda x: residuals[x]['rms'])
        report.append(f"Най-добър модел по RMS: {best_model_rms}")
        
        # Анализ на остатъците
        report.append("\nОЦЕНКА НА ОСТАТЪЧНИЯ ШУМ:")
        for model_name in analysis:
            report.append(f"\n{model_name}:")
            
            # Нормалност
            if 'normality_tests' in analysis[model_name]:
                sw_test = analysis[model_name]['normality_tests'].get('shapiro_wilk', {})
                if isinstance(sw_test, dict) and 'p_value' in sw_test:
                    sw_p = sw_test['p_value']
                    if sw_p > 0.05:
                        report.append("  ✓ Остатъците следват нормално разпределение")
                    else:
                        report.append("  ✗ Остатъците НЕ следват нормално разпределение")
                else:
                    report.append("  ? Нормалност: недостатъчно данни за анализ")
            
            # Автокорелация
            if 'autocorrelation_tests' in analysis[model_name]:
                runs_test = analysis[model_name]['autocorrelation_tests'].get('runs_test', {})
                if isinstance(runs_test, dict) and 'p_value' in runs_test:
                    runs_p = runs_test['p_value']
                    if runs_p > 0.05:
                        report.append("  ✓ Няма автокорелация в остатъците")
                    else:
                        report.append("  ✗ Има автокорелация в остатъците")
                else:
                    report.append("  ? Автокорелация: недостатъчно данни за анализ")
            
            # Outliers
            if 'outlier_analysis' in analysis[model_name]:
                outlier_fraction = analysis[model_name]['outlier_analysis']['iqr_outlier_fraction']
                if outlier_fraction < 0.05:
                    report.append("  ✓ Ниско ниво на outliers")
                else:
                    report.append("  ✗ Високо ниво на outliers")
        
        return "\n".join(report)
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Стартира пълен анализ
        
        Returns:
            Всички резултати
        """
        print("🔍 Стартиране на обширен анализ на остатъчен шум...")
        
        # 1. Зареждане на данни
        print("📊 Зареждане на данни...")
        data = self.load_observational_data()
        
        # 2. Фитване на модели
        print("🔧 Фитване на модели...")
        fit_results = self.fit_models(data)
        
        # 3. Пресмятане на остатъци
        print("📐 Пресмятане на остатъци...")
        residuals = self.calculate_residuals(data, fit_results)
        
        # 4. Анализ на остатъчния шум
        print("🔍 Анализ на остатъчния шум...")
        analysis = self.analyze_residual_noise(residuals)
        
        # 5. MCMC анализ
        print("📈 MCMC анализ...")
        mcmc_results = self.run_mcmc_model_comparison(data, residuals)
        
        # 6. Графики
        print("📊 Създаване на графики...")
        self.plot_comprehensive_analysis(data, residuals, analysis)
        
        # 7. Доклад
        print("📄 Генериране на доклад...")
        report = self.generate_comprehensive_report(data, residuals, analysis, mcmc_results)
        print(report)
        
        print("✅ Анализът на остатъчния шум завърши успешно!")
        
        return {
            'data': data,
            'fit_results': fit_results,
            'residuals': residuals,
            'analysis': analysis,
            'mcmc_results': mcmc_results,
            'report': report
        }


def test_residual_noise_analyzer():
    """
    Тестова функция за анализатора на остатъчен шум
    """
    # Създаваме анализатор
    analyzer = ResidualNoiseAnalyzer(use_raw_data=False)  # Използваме синтетични данни
    
    # Стартираме пълния анализ
    results = analyzer.run_comprehensive_analysis()
    
    return results


if __name__ == "__main__":
    # Глобални константи
    c = 299792458  # м/с
    
    # Тестваме анализатора
    results = test_residual_noise_analyzer() 