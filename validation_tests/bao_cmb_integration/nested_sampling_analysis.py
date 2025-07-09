#!/usr/bin/env python3
"""
Nested Sampling анализ за модел селекция и Bayesian evidence

Този модул предоставя:
1. Nested sampling с dynesty
2. Модел селекция ΛCDM vs No-Λ
3. Bayesian evidence изчисление
4. Information criteria (AIC, BIC, DIC)
5. Posterior probability интеграция
6. Model comparison и odds ratios
"""

import numpy as np
import matplotlib.pyplot as plt
from dynesty import DynamicNestedSampler, NestedSampler
from dynesty.plotting import runplot, traceplot, cornerplot
from dynesty.utils import resample_equal
import corner
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
import logging
import warnings
import time
from multiprocessing import Pool

# Импортиране на нашите модули
from mcmc_analysis import MCMCAnalysis
from observational_data import BAOObservationalData, CMBObservationalData, LikelihoodFunctions
from no_lambda_cosmology import NoLambdaCosmology

# Настройка на логирането
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NestedSamplingAnalysis:
    """
    Клас за nested sampling анализ на космологични модели
    
    Предоставя функции за:
    - Bayesian evidence изчисление
    - Модел селекция
    - Information criteria
    - Model comparison
    """
    
    def __init__(self, 
                 parameter_names: List[str] = None,
                 parameter_ranges: Dict[str, Tuple[float, float]] = None,
                 nlive: int = 1000):
        """
        Инициализация на nested sampling анализа
        
        Args:
            parameter_names: Имена на параметрите
            parameter_ranges: Диапазони на параметрите
            nlive: Брой живи точки за nested sampling
        """
        
        # Зареждане на наблюдателни данни
        self.bao_data = BAOObservationalData()
        self.cmb_data = CMBObservationalData()
        self.likelihood_func = LikelihoodFunctions(self.bao_data, self.cmb_data)
        
        # Параметри на модела
        if parameter_names is None:
            parameter_names = ['H0', 'Omega_m', 'Omega_b', 'epsilon_bao', 'epsilon_cmb']
        
        self.parameter_names = parameter_names
        self.n_params = len(parameter_names)
        
        # Диапазони на параметрите
        if parameter_ranges is None:
            parameter_ranges = {
                'H0': (60.0, 80.0),
                'Omega_m': (0.20, 0.50),
                'Omega_b': (0.030, 0.070),
                'epsilon_bao': (0.0, 0.10),
                'epsilon_cmb': (0.0, 0.08),
                'alpha': (0.5, 2.0),
                'beta': (0.0, 0.5),
                'gamma': (0.1, 1.0),
                'delta': (0.01, 0.20),
                'angular_strength': (0.1, 1.0)
            }
        
        self.parameter_ranges = parameter_ranges
        
        # Nested sampling настройки
        self.nlive = nlive
        self.dlogz = 0.01
        self.maxiter = 10000
        self.maxcall = 1000000
        
        # Резултати
        self.sampler = None
        self.results = None
        self.log_evidence = None
        self.log_evidence_err = None
        self.posterior_samples = None
        
        logger.info(f"Инициализиран nested sampling анализ с {self.n_params} параметра")
        logger.info(f"Параметри: {self.parameter_names}")
        logger.info(f"Nlive: {self.nlive}")
    
    def ptform(self, u: np.ndarray) -> np.ndarray:
        """
        Prior transform функция за nested sampling
        
        Args:
            u: Uniform random values [0,1]
            
        Returns:
            Transformed parameters
        """
        x = np.zeros(self.n_params)
        
        for i, param_name in enumerate(self.parameter_names):
            if param_name in self.parameter_ranges:
                param_min, param_max = self.parameter_ranges[param_name]
                x[i] = param_min + u[i] * (param_max - param_min)
            else:
                x[i] = u[i]  # Default [0,1] range
        
        return x
    
    def loglike(self, params: np.ndarray) -> float:
        """
        Log-likelihood функция за nested sampling
        
        Args:
            params: Параметри на модела
            
        Returns:
            Log-likelihood стойност
        """
        try:
            # Създаване на параметрични речници
            param_dict = {}
            for i, param_name in enumerate(self.parameter_names):
                param_dict[param_name] = params[i]
            
            # Задаване на default стойности
            default_params = {
                'H0': 67.4,
                'Omega_m': 0.315,
                'Omega_b': 0.049,
                'Omega_cdm': 0.266,
                'Omega_r': 8.24e-5,
                'epsilon_bao': 0.02,
                'epsilon_cmb': 0.015,
                'alpha': 1.2,
                'beta': 0.0,
                'gamma': 0.4,
                'delta': 0.08,
                'angular_strength': 0.6
            }
            
            # Обновяване с параметрите
            for key, value in param_dict.items():
                default_params[key] = value
            
            # Проверка на консистентност
            if default_params['Omega_m'] < default_params['Omega_b']:
                return -np.inf
            
            # Създаване на модел
            cosmo = NoLambdaCosmology(**default_params)
            
            # Изчисляване на предсказанията
            model_predictions = self._calculate_model_predictions(cosmo)
            
            # Изчисляване на likelihood
            log_like = self.likelihood_func.combined_likelihood(model_predictions)
            
            return log_like
            
        except Exception as e:
            logger.debug(f"Грешка в likelihood: {e}")
            return -np.inf
    
    def _calculate_model_predictions(self, cosmo: NoLambdaCosmology) -> Dict:
        """
        Изчисляване на предсказанията на модела
        
        Args:
            cosmo: Космологичен модел
            
        Returns:
            Предсказания на модела
        """
        # Получаване на BAO данни
        bao_combined = self.bao_data.get_combined_data()
        z_bao = bao_combined['redshifts']
        
        # Константа за скоростта на светлината
        c_km_s = 299792.458  # km/s
        
        # Изчисляване на D_V/r_s за BAO
        r_s = cosmo.sound_horizon_scale()
        DV_rs_model = []
        
        for z in z_bao:
            # Изчисляване на D_V(z)
            D_A = cosmo.angular_diameter_distance(z)
            D_H = c_km_s / (cosmo.hubble_function(z) * 1000)  # Hubble distance
            D_V = (z * D_A**2 * D_H)**(1/3)  # Dilation distance
            
            DV_rs_model.append(D_V / r_s)
        
        # Изчисляване на CMB предсказания
        l_peaks_model = []
        for i in range(1, 4):  # Първи 3 пика
            theta_s = cosmo.cmb_angular_scale()
            l_peak = i * np.pi / theta_s
            l_peaks_model.append(l_peak)
        
        theta_s_model = cosmo.cmb_angular_scale()
        
        return {
            'DV_rs': np.array(DV_rs_model),
            'l_peaks': np.array(l_peaks_model),
            'theta_s': theta_s_model
        }
    
    def run_nested_sampling(self, 
                          nlive: int = None,
                          dlogz: float = None,
                          dynamic: bool = True,
                          progress: bool = True) -> None:
        """
        Изпълнение на nested sampling
        
        Args:
            nlive: Брой живи точки
            dlogz: Accuracy в log-evidence
            dynamic: Използване на dynamic nested sampling
            progress: Показване на прогрес
        """
        
        # Използване на default стойности
        if nlive is None:
            nlive = self.nlive
        if dlogz is None:
            dlogz = self.dlogz
        
        logger.info(f"Стартиране на nested sampling:")
        logger.info(f"  Nlive: {nlive}")
        logger.info(f"  dlogz: {dlogz}")
        logger.info(f"  Dynamic: {dynamic}")
        
        start_time = time.time()
        
        # Избор на sampler
        if dynamic:
            sampler = DynamicNestedSampler(
                self.loglike,
                self.ptform,
                self.n_params,
                nlive=nlive
            )
        else:
            sampler = NestedSampler(
                self.loglike,
                self.ptform,
                self.n_params,
                nlive=nlive
            )
        
        # Изпълнение на sampling
        logger.info("Изпълнение на nested sampling...")
        
        if dynamic:
            sampler.run_nested(print_progress=progress)
        else:
            sampler.run_nested(print_progress=progress)
        
        # Съхранение на резултатите
        self.sampler = sampler
        self.results = sampler.results
        
        # Извличане на evidence
        self.log_evidence = self.results.logz[-1]
        self.log_evidence_err = self.results.logzerr[-1]
        
        # Posterior samples
        self.posterior_samples = resample_equal(
            self.results.samples,
            self.results.logwt
        )
        
        end_time = time.time()
        runtime = end_time - start_time
        
        logger.info(f"Nested sampling завършен за {runtime:.1f} секунди")
        logger.info(f"Log-evidence: {self.log_evidence:.3f} ± {self.log_evidence_err:.3f}")
        logger.info(f"Posterior samples: {len(self.posterior_samples)}")
        
        # Анализ на резултатите
        self._analyze_results()
    
    def _analyze_results(self) -> None:
        """Анализ на nested sampling резултатите"""
        
        if self.results is None:
            logger.warning("Няма резултати за анализ")
            return
        
        # Информационни критерии
        n_data = len(self.bao_data.get_combined_data()['redshifts']) + 4  # BAO + CMB
        
        # Най-добър likelihood
        best_log_like = np.max(self.results.logl)
        
        # AIC и BIC
        aic = 2 * self.n_params - 2 * best_log_like
        bic = np.log(n_data) * self.n_params - 2 * best_log_like
        
        # DIC (Deviance Information Criterion)
        posterior_mean_loglike = np.mean(self.results.logl)
        effective_params = 2 * (best_log_like - posterior_mean_loglike)
        dic = -2 * posterior_mean_loglike + 2 * effective_params
        
        # Съхранение на информацията
        self.info_criteria = {
            'log_evidence': self.log_evidence,
            'log_evidence_err': self.log_evidence_err,
            'best_log_likelihood': best_log_like,
            'aic': aic,
            'bic': bic,
            'dic': dic,
            'effective_params': effective_params,
            'n_parameters': self.n_params,
            'n_data': n_data
        }
        
        # Параметрични статистики
        self.param_stats = {}
        
        for i, param_name in enumerate(self.parameter_names):
            samples_param = self.posterior_samples[:, i]
            
            # Средна стойност
            mean_val = np.mean(samples_param)
            
            # Несигурности (68% credible interval)
            percentiles = np.percentile(samples_param, [16, 50, 84])
            lower_err = percentiles[1] - percentiles[0]
            upper_err = percentiles[2] - percentiles[1]
            
            self.param_stats[param_name] = {
                'mean': mean_val,
                'median': percentiles[1],
                'lower_err': lower_err,
                'upper_err': upper_err,
                'std': np.std(samples_param)
            }
        
        logger.info("Анализ на nested sampling резултатите завършен")
    
    def plot_run(self, save_path: str = None) -> None:
        """
        Графики на nested sampling run
        
        Args:
            save_path: Път за записване
        """
        if self.results is None:
            logger.warning("Няма резултати за визуализация")
            return
        
        fig, axes = runplot(self.results, color='blue')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_trace(self, save_path: str = None) -> None:
        """
        Trace plots за nested sampling
        
        Args:
            save_path: Път за записване
        """
        if self.results is None:
            logger.warning("Няма резултати за визуализация")
            return
        
        fig, axes = traceplot(self.results, labels=self.parameter_names)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_corner(self, save_path: str = None, truth_values: Dict = None) -> None:
        """
        Corner plot за nested sampling резултатите
        
        Args:
            save_path: Път за записване
            truth_values: Истински стойности
        """
        if self.posterior_samples is None:
            logger.warning("Няма posterior samples за corner plot")
            return
        
        # Подготовка на truth values
        truths = None
        if truth_values:
            truths = []
            for param_name in self.parameter_names:
                if param_name in truth_values:
                    truths.append(truth_values[param_name])
                else:
                    truths.append(None)
        
        # Създаване на corner plot
        fig = corner.corner(
            self.posterior_samples,
            labels=self.parameter_names,
            truths=truths,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def calculate_bayes_factor(self, other_model: 'NestedSamplingAnalysis') -> Dict:
        """
        Изчисляване на Bayes factor между два модела
        
        Args:
            other_model: Друг модел за сравнение
            
        Returns:
            Bayes factor анализ
        """
        if self.log_evidence is None or other_model.log_evidence is None:
            logger.warning("Едно или двете модели няма evidence")
            return {}
        
        # Bayes factor
        log_bayes_factor = self.log_evidence - other_model.log_evidence
        bayes_factor = np.exp(log_bayes_factor)
        
        # Несигурност в Bayes factor
        log_bf_err = np.sqrt(self.log_evidence_err**2 + other_model.log_evidence_err**2)
        
        # Jeffreys' scale интерпретация
        if np.abs(log_bayes_factor) < 1.0:
            interpretation = "Inconclusive"
        elif np.abs(log_bayes_factor) < 2.5:
            interpretation = "Weak evidence"
        elif np.abs(log_bayes_factor) < 5.0:
            interpretation = "Moderate evidence"
        else:
            interpretation = "Strong evidence"
        
        preferred_model = "Model 1 (this)" if log_bayes_factor > 0 else "Model 2 (other)"
        
        return {
            'log_bayes_factor': log_bayes_factor,
            'log_bayes_factor_err': log_bf_err,
            'bayes_factor': bayes_factor,
            'preferred_model': preferred_model,
            'interpretation': interpretation,
            'model1_log_evidence': self.log_evidence,
            'model2_log_evidence': other_model.log_evidence
        }
    
    def summary(self) -> None:
        """Резюме на nested sampling анализа"""
        
        print("🎯 NESTED SAMPLING АНАЛИЗ РЕЗУЛТАТИ")
        print("=" * 70)
        
        if self.results is None:
            print("Няма анализирани резултати")
            return
        
        # Evidence информация
        print(f"\n📊 BAYESIAN EVIDENCE:")
        print(f"  Log-evidence: {self.log_evidence:.3f} ± {self.log_evidence_err:.3f}")
        print(f"  Evidence: {np.exp(self.log_evidence):.2e}")
        
        # Information criteria
        if hasattr(self, 'info_criteria'):
            info = self.info_criteria
            print(f"\n📈 INFORMATION CRITERIA:")
            print(f"  AIC: {info['aic']:.2f}")
            print(f"  BIC: {info['bic']:.2f}")
            print(f"  DIC: {info['dic']:.2f}")
            print(f"  Effective parameters: {info['effective_params']:.1f}")
            print(f"  Best log-likelihood: {info['best_log_likelihood']:.2f}")
        
        # Параметрични оценки
        if hasattr(self, 'param_stats'):
            print(f"\n🔍 ПАРАМЕТРИЧНИ ОЦЕНКИ:")
            print(f"{'Параметър':<15} {'Средна':<10} {'Медиана':<10} {'±Долна':<10} {'±Горна':<10}")
            print("-" * 70)
            
            for param_name in self.parameter_names:
                if param_name in self.param_stats:
                    stats = self.param_stats[param_name]
                    print(f"{param_name:<15} {stats['mean']:<10.4f} {stats['median']:<10.4f} "
                          f"{stats['lower_err']:<10.4f} {stats['upper_err']:<10.4f}")
        
        # Sampling информация
        print(f"\n⚙️  SAMPLING INFORMATION:")
        print(f"  Nlive: {self.nlive}")
        print(f"  Samples: {len(self.posterior_samples) if self.posterior_samples is not None else 'N/A'}")
        print(f"  Iterations: {len(self.results.logz)}")
        print(f"  Calls: {self.results.ncall}")


def compare_models(model1: NestedSamplingAnalysis, 
                  model2: NestedSamplingAnalysis,
                  model1_name: str = "Model 1",
                  model2_name: str = "Model 2") -> None:
    """
    Сравнение между два модела
    
    Args:
        model1: Първи модел
        model2: Втори модел
        model1_name: Име на първия модел
        model2_name: Име на втория модел
    """
    
    print("🔄 МОДЕЛ СРАВНЕНИЕ")
    print("=" * 70)
    
    # Bayes factor анализ
    bf_analysis = model1.calculate_bayes_factor(model2)
    
    if bf_analysis:
        print(f"\n📊 BAYES FACTOR АНАЛИЗ:")
        print(f"  {model1_name} vs {model2_name}")
        print(f"  Log Bayes Factor: {bf_analysis['log_bayes_factor']:.3f} ± {bf_analysis['log_bayes_factor_err']:.3f}")
        print(f"  Bayes Factor: {bf_analysis['bayes_factor']:.2e}")
        print(f"  Предпочитан модел: {bf_analysis['preferred_model']}")
        print(f"  Интерпретация: {bf_analysis['interpretation']}")
        
        print(f"\n📈 EVIDENCE COMPARISON:")
        print(f"  {model1_name} log-evidence: {bf_analysis['model1_log_evidence']:.3f}")
        print(f"  {model2_name} log-evidence: {bf_analysis['model2_log_evidence']:.3f}")
    
    # Information criteria сравнение
    if hasattr(model1, 'info_criteria') and hasattr(model2, 'info_criteria'):
        print(f"\n📊 INFORMATION CRITERIA COMPARISON:")
        print(f"{'Criterion':<15} {model1_name:<15} {model2_name:<15} {'Difference':<15}")
        print("-" * 70)
        
        for criterion in ['aic', 'bic', 'dic']:
            val1 = model1.info_criteria[criterion]
            val2 = model2.info_criteria[criterion]
            diff = val1 - val2
            print(f"{criterion.upper():<15} {val1:<15.2f} {val2:<15.2f} {diff:<15.2f}")


def test_nested_sampling():
    """Тест на nested sampling анализа"""
    
    print("🧪 ТЕСТ НА NESTED SAMPLING АНАЛИЗ")
    print("=" * 70)
    
    # Създаване на nested sampling анализатор
    ns = NestedSamplingAnalysis(
        parameter_names=['H0', 'Omega_m', 'epsilon_bao'],
        nlive=100  # Малко за тест
    )
    
    print("Стартиране на тестов nested sampling...")
    ns.run_nested_sampling(nlive=100, dlogz=0.1, progress=True)
    
    # Показване на резултатите
    ns.summary()
    
    # Графики
    ns.plot_run(save_path='nested_sampling_run_test.png')
    ns.plot_trace(save_path='nested_sampling_trace_test.png')
    ns.plot_corner(save_path='nested_sampling_corner_test.png')
    
    print("\n✅ Nested sampling тестът завърши успешно!")


if __name__ == "__main__":
    test_nested_sampling() 