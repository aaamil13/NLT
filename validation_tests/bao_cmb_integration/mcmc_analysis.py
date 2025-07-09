#!/usr/bin/env python3
"""
MCMC анализ за параметрична калибрация на No-Λ модела

Този модул предоставя:
1. MCMC sampling с emcee
2. Priors за космологични параметри
3. Posterior анализ и маргинализации
4. Corner plots за параметрични корелации
5. Convergence диагностики
6. Параметрични несигурности
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from scipy import stats
from scipy.optimize import minimize
from scipy.constants import c
from typing import Dict, List, Tuple, Optional, Callable
import logging
import warnings
from multiprocessing import Pool

# Импортиране на нашите модули
from no_lambda_cosmology import NoLambdaCosmology
from observational_data import BAOObservationalData, CMBObservationalData, LikelihoodFunctions

# Настройка на стиловете
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Използваме default стил

# Настройка на логирането
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константа за скоростта на светлината в km/s
c_km_s = 299792.458  # km/s


class MCMCAnalysis:
    """
    Клас за MCMC анализ на No-Λ модела
    
    Предоставя функции за:
    - Параметрична калибрация
    - Байесов анализ
    - Posterior маргинализации
    - Статистически тестове
    """
    
    def __init__(self, 
                 parameter_names: List[str] = None,
                 parameter_ranges: Dict[str, Tuple[float, float]] = None,
                 use_anisotropy: bool = True):
        """
        Инициализация на MCMC анализа
        
        Args:
            parameter_names: Имена на параметрите за калибрация
            parameter_ranges: Диапазони на параметрите
            use_anisotropy: Дали да се използва анизотропия
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
        self.use_anisotropy = use_anisotropy
        
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
        
        # MCMC настройки
        self.n_walkers = 50
        self.n_burn = 1000
        self.n_samples = 5000
        self.n_threads = 4
        
        # Резултати
        self.sampler = None
        self.samples = None
        self.best_params = None
        self.param_uncertainties = None
        
        logger.info(f"Инициализиран MCMC анализ с {self.n_params} параметра")
        logger.info(f"Параметри: {self.parameter_names}")
    
    def log_prior(self, params: np.ndarray) -> float:
        """
        Logarithmic prior вероятност
        
        Args:
            params: Параметри на модела
            
        Returns:
            Log-prior стойност
        """
        log_prior_value = 0.0
        
        for i, param_name in enumerate(self.parameter_names):
            if param_name in self.parameter_ranges:
                param_min, param_max = self.parameter_ranges[param_name]
                
                if param_min <= params[i] <= param_max:
                    # Uniform prior
                    log_prior_value += -np.log(param_max - param_min)
                else:
                    return -np.inf
        
        return log_prior_value
    
    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Logarithmic likelihood функция
        
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
            
            # Задаване на default стойности за липсващи параметри
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
            
            # Обновяване с параметрите от MCMC
            for key, value in param_dict.items():
                default_params[key] = value
            
            # Проверка на консистентност
            if default_params['Omega_m'] < default_params['Omega_b']:
                return -np.inf
            
            # Създаване на модел
            cosmo = NoLambdaCosmology(**default_params)
            
            # Изчисляване на предсказанията на модела
            model_predictions = self._calculate_model_predictions(cosmo)
            
            # Изчисляване на likelihood
            log_like = self.likelihood_func.combined_likelihood(model_predictions)
            
            return log_like
            
        except Exception as e:
            logger.warning(f"Грешка в likelihood: {e}")
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
    
    def log_posterior(self, params: np.ndarray) -> float:
        """
        Logarithmic posterior вероятност
        
        Args:
            params: Параметри на модела
            
        Returns:
            Log-posterior стойност
        """
        log_prior_val = self.log_prior(params)
        if not np.isfinite(log_prior_val):
            return -np.inf
        
        log_like_val = self.log_likelihood(params)
        if not np.isfinite(log_like_val):
            return -np.inf
        
        return log_prior_val + log_like_val
    
    def find_best_fit(self, initial_guess: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """
        Намиране на най-добрия fit с оптимизация
        
        Args:
            initial_guess: Начални стойности
            
        Returns:
            Най-добри параметри и negative log-likelihood
        """
        if initial_guess is None:
            # Взимаме средните стойности от ranges
            initial_guess = []
            for param_name in self.parameter_names:
                param_min, param_max = self.parameter_ranges[param_name]
                initial_guess.append((param_min + param_max) / 2)
            initial_guess = np.array(initial_guess)
        
        def negative_log_posterior(params):
            return -self.log_posterior(params)
        
        # Оптимизация
        result = minimize(negative_log_posterior, initial_guess, method='L-BFGS-B')
        
        if result.success:
            logger.info(f"Оптимизацията успешна: {result.fun:.3f}")
            self.best_params = result.x
            return result.x, result.fun
        else:
            logger.warning(f"Оптимизацията неуспешна: {result.message}")
            return initial_guess, negative_log_posterior(initial_guess)
    
    def run_mcmc(self, 
                 n_walkers: int = None,
                 n_burn: int = None,
                 n_samples: int = None,
                 initial_params: np.ndarray = None,
                 progress: bool = True) -> None:
        """
        Изпълнение на MCMC sampling
        
        Args:
            n_walkers: Брой walkers
            n_burn: Брой burn-in стъпки
            n_samples: Брой samples
            initial_params: Начални параметри
            progress: Показване на прогреса
        """
        
        # Използване на default стойности
        if n_walkers is None:
            n_walkers = self.n_walkers
        if n_burn is None:
            n_burn = self.n_burn
        if n_samples is None:
            n_samples = self.n_samples
        
        # Намиране на най-добрия fit за начални стойности
        if initial_params is None:
            logger.info("Търсене на най-добрия fit...")
            initial_params, _ = self.find_best_fit()
        
        # Инициализиране на walkers около най-добрия fit
        pos = initial_params + 1e-4 * np.random.randn(n_walkers, self.n_params)
        
        # Проверка на priors
        for i in range(n_walkers):
            while not np.isfinite(self.log_prior(pos[i])):
                pos[i] = initial_params + 1e-4 * np.random.randn(self.n_params)
        
        # Създаване на sampler
        logger.info(f"Стартиране на MCMC: {n_walkers} walkers, {n_burn} burn-in, {n_samples} samples")
        
        self.sampler = emcee.EnsembleSampler(
            n_walkers, 
            self.n_params, 
            self.log_posterior
        )
        
        # Burn-in
        logger.info("Изпълнение на burn-in...")
        pos, _, _ = self.sampler.run_mcmc(pos, n_burn, progress=progress)
        self.sampler.reset()
        
        # Production run
        logger.info("Изпълнение на production sampling...")
        self.sampler.run_mcmc(pos, n_samples, progress=progress)
        
        # Извличане на samples
        self.samples = self.sampler.get_chain(discard=0, thin=1, flat=True)
        
        logger.info(f"MCMC завършен: {self.samples.shape[0]} samples")
        
        # Анализ на резултатите
        self._analyze_results()
    
    def _analyze_results(self) -> None:
        """Анализ на MCMC резултатите"""
        
        if self.samples is None:
            logger.warning("Няма samples за анализ")
            return
        
        # Средни стойности и несигурности
        self.param_uncertainties = {}
        
        for i, param_name in enumerate(self.parameter_names):
            samples_param = self.samples[:, i]
            
            # Средна стойност
            mean_val = np.mean(samples_param)
            
            # Несигурности (68% credible interval)
            percentiles = np.percentile(samples_param, [16, 50, 84])
            lower_err = percentiles[1] - percentiles[0]
            upper_err = percentiles[2] - percentiles[1]
            
            self.param_uncertainties[param_name] = {
                'mean': mean_val,
                'median': percentiles[1],
                'lower_err': lower_err,
                'upper_err': upper_err,
                'std': np.std(samples_param)
            }
        
        # Автокорелационен анализ
        try:
            autocorr_times = self.sampler.get_autocorr_time()
            logger.info(f"Автокорелационни времена: {autocorr_times}")
        except Exception as e:
            logger.warning(f"Не може да се изчисли автокорелационното време: {e}")
        
        # Acceptance fraction
        acceptance_fraction = np.mean(self.sampler.acceptance_fraction)
        logger.info(f"Acceptance fraction: {acceptance_fraction:.3f}")
    
    def plot_chains(self, save_path: str = None) -> None:
        """
        Графики на MCMC chains
        
        Args:
            save_path: Път за записване
        """
        if self.sampler is None:
            logger.warning("Няма sampler за визуализация")
            return
        
        fig, axes = plt.subplots(self.n_params, 1, figsize=(12, 2*self.n_params))
        
        if self.n_params == 1:
            axes = [axes]
        
        samples = self.sampler.get_chain()
        
        for i, param_name in enumerate(self.parameter_names):
            ax = axes[i]
            ax.plot(samples[:, :, i], alpha=0.3)
            ax.set_ylabel(param_name)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Стъпка')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_corner(self, save_path: str = None, truth_values: Dict = None) -> None:
        """
        Corner plot за параметричните корелации
        
        Args:
            save_path: Път за записване
            truth_values: Истински стойности на параметрите
        """
        if self.samples is None:
            logger.warning("Няма samples за corner plot")
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
            self.samples,
            labels=self.parameter_names,
            truths=truths,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def calculate_evidence(self) -> Dict:
        """
        Изчисляване на Bayesian evidence (приближение)
        
        Returns:
            Речник с evidence оценки
        """
        if self.samples is None:
            logger.warning("Няма samples за evidence изчисление")
            return {}
        
        # Harmonic mean estimator (приближение)
        log_likelihoods = []
        for i in range(len(self.samples)):
            log_like = self.log_likelihood(self.samples[i])
            if np.isfinite(log_like):
                log_likelihoods.append(log_like)
        
        if not log_likelihoods:
            return {}
        
        log_likelihoods = np.array(log_likelihoods)
        
        # Harmonic mean
        max_log_like = np.max(log_likelihoods)
        shifted_log_likes = log_likelihoods - max_log_like
        
        harmonic_mean = -np.log(np.mean(np.exp(-shifted_log_likes))) + max_log_like
        
        # Информационни критерии
        n_data = len(self.bao_data.get_combined_data()['redshifts']) + 4  # BAO + CMB
        
        # Най-добър likelihood
        best_log_like = np.max(log_likelihoods)
        
        # AIC и BIC
        aic = 2 * self.n_params - 2 * best_log_like
        bic = np.log(n_data) * self.n_params - 2 * best_log_like
        
        return {
            'log_evidence_harmonic': harmonic_mean,
            'best_log_likelihood': best_log_like,
            'aic': aic,
            'bic': bic,
            'n_parameters': self.n_params,
            'n_data': n_data
        }
    
    def summary(self) -> None:
        """Резюме на MCMC анализа"""
        
        print("📊 MCMC АНАЛИЗ РЕЗУЛТАТИ")
        print("=" * 70)
        
        if self.param_uncertainties is None:
            print("Няма анализирани резултати")
            return
        
        print(f"\n🔍 ПАРАМЕТРИЧНИ ОЦЕНКИ:")
        print(f"{'Параметър':<15} {'Средна':<10} {'Медиана':<10} {'±Долна':<10} {'±Горна':<10}")
        print("-" * 70)
        
        for param_name in self.parameter_names:
            if param_name in self.param_uncertainties:
                stats = self.param_uncertainties[param_name]
                print(f"{param_name:<15} {stats['mean']:<10.4f} {stats['median']:<10.4f} "
                      f"{stats['lower_err']:<10.4f} {stats['upper_err']:<10.4f}")
        
        # Evidence анализ
        evidence = self.calculate_evidence()
        if evidence:
            print(f"\n📈 БАЙЕСОВ АНАЛИЗ:")
            print(f"  Log-evidence (harmonic): {evidence['log_evidence_harmonic']:.2f}")
            print(f"  Best log-likelihood: {evidence['best_log_likelihood']:.2f}")
            print(f"  AIC: {evidence['aic']:.2f}")
            print(f"  BIC: {evidence['bic']:.2f}")
        
        # Автокорелация
        if self.sampler:
            try:
                autocorr_times = self.sampler.get_autocorr_time()
                print(f"\n⏱️  АВТОКОРЕЛАЦИОННИ ВРЕМЕНА:")
                for i, param_name in enumerate(self.parameter_names):
                    print(f"  {param_name}: {autocorr_times[i]:.1f}")
            except:
                pass
            
            acceptance = np.mean(self.sampler.acceptance_fraction)
            print(f"\n✅ ACCEPTANCE FRACTION: {acceptance:.3f}")


def test_mcmc_analysis():
    """Тест на MCMC анализа"""
    
    print("🧪 ТЕСТ НА MCMC АНАЛИЗ")
    print("=" * 70)
    
    # Създаване на MCMC анализатор
    mcmc = MCMCAnalysis(
        parameter_names=['H0', 'Omega_m', 'epsilon_bao'],
        use_anisotropy=True
    )
    
    # Кратък тест run
    mcmc.n_walkers = 20
    mcmc.n_burn = 50
    mcmc.n_samples = 200
    
    print("Стартиране на тестов MCMC...")
    mcmc.run_mcmc(progress=True)
    
    # Показване на резултатите
    mcmc.summary()
    
    # Графики
    mcmc.plot_chains(save_path='mcmc_chains_test.png')
    mcmc.plot_corner(save_path='mcmc_corner_test.png')
    
    print("\n✅ MCMC тестът завърши успешно!")


if __name__ == "__main__":
    test_mcmc_analysis() 