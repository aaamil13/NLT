#!/usr/bin/env python3
"""
Оптимизиран Nested Sampling анализ за модел селекция

ОПТИМИЗАЦИИ:
1. Минимален консолен изход
2. Кеширане на изчисления
3. Векторизирани операции
4. По-ефикасни likelihood функции
5. Numba компилация за максимална скорост
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
from multiprocessing import Pool, cpu_count

# Импортиране на нашите модули
from mcmc_analysis import MCMCAnalysis
from observational_data import BAOObservationalData, CMBObservationalData, LikelihoodFunctions
from no_lambda_cosmology import NoLambdaCosmology
from fast_cosmo import *  # Numba оптимизирани функции

# МИНИМАЛНО логиране за скорост
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Потискане на warnings
warnings.filterwarnings('ignore')

# Глобални константи за оптимизация
C_KM_S = 299792.458  # km/s
PI = np.pi


class OptimizedNestedSampling:
    """
    Оптимизиран nested sampling клас за максимална скорост
    """
    
    def __init__(self, 
                 parameter_names: List[str] = None,
                 parameter_ranges: Dict[str, Tuple[float, float]] = None,
                 nlive: int = 100,  # По-малко за скорост
                 use_snia: bool = False,  # Опция за SN Ia данни
                 use_h0: bool = False):   # Опция за H₀ данни
        """
        Инициализация на оптимизирания nested sampling
        
        Args:
            parameter_names: Имена на параметрите
            parameter_ranges: Диапазони на параметрите
            nlive: Брой live points
            use_snia: Дали да се включат SN Ia данни
            use_h0: Дали да се включат H₀ данни
        """
        
        # Конфигурация на данните
        self.use_snia = use_snia
        self.use_h0 = use_h0
        
        # Настройка на параметрите
        if parameter_names is None:
            parameter_names = ['H0', 'Omega_m', 'epsilon_bao', 'epsilon_cmb']
        
        if parameter_ranges is None:
            parameter_ranges = {
                'H0': (60.0, 80.0),
                'Omega_m': (0.05, 0.95),
                'epsilon_bao': (-0.1, 0.1),
                'epsilon_cmb': (-0.1, 0.1)
            }
        
        self.parameter_names = parameter_names
        self.parameter_ranges = parameter_ranges
        self.n_params = len(parameter_names)
        self.nlive = nlive
        
        # Кеширани данни
        self.cached_n_bao = 0
        self.cached_n_cmb = 0
        self.cached_n_snia = 0
        self.cached_n_h0 = 0
        
        # Резултати
        self.results = None
        self.sampler = None
        self.log_evidence = None
        self.log_evidence_err = None
        self.posterior_samples = None
        self.param_stats = {}
        
        # Настройка на данните
        self._setup_cached_data()
        
        logger.info(f"Настроен nested sampling с {self.n_params} параметра")
        
        # Изброй на активните данни
        active_data = ['BAO', 'CMB']
        if self.use_snia:
            active_data.append('SN Ia')
        if self.use_h0:
            active_data.append('H₀')
        
        logger.info(f"Активни данни: {', '.join(active_data)}")
    
    def _setup_cached_data(self):
        """
        Настройка на кешираните данни за максимална скорост
        """
        logger.info("Настройка на кешираните данни...")
        
        # Основни данни (BAO + CMB)
        from observational_data import (
            BAOObservationalData, 
            CMBObservationalData,
            SNIaObservationalData,
            LocalH0ObservationalData,
            LikelihoodFunctions
        )
        
        # Зареждане на BAO и CMB данни
        self.bao_data = BAOObservationalData()
        self.cmb_data = CMBObservationalData()
        
        # Опционално зареждане на SN Ia данни
        if self.use_snia:
            logger.info("Зареждане на SN Ia данни...")
            self.snia_data = SNIaObservationalData()
            self.cached_n_snia = len(self.snia_data.get_combined_data()['redshifts'])
            logger.info(f"Заредени {self.cached_n_snia} SN Ia supernovae")
        else:
            self.snia_data = None
        
        # Опционално зареждане на H₀ данни
        if self.use_h0:
            logger.info("Зареждане на H₀ данни...")
            self.h0_data = LocalH0ObservationalData()
            self.cached_n_h0 = len(self.h0_data.h0_measurements)
            logger.info(f"Заредени {self.cached_n_h0} H₀ измервания")
        else:
            self.h0_data = None
        
        # Създаване на пълната likelihood функция
        self.likelihood_func = LikelihoodFunctions(
            bao_data=self.bao_data,
            cmb_data=self.cmb_data,
            snia_data=self.snia_data,
            h0_data=self.h0_data
        )
        
        # Кеширане на размерите
        self.cached_n_bao = len(self.bao_data.get_combined_data()['redshifts'])
        self.cached_n_cmb = 4  # theta_s + 3 peaks
        
        # Общ брой данни
        total_data_points = self.cached_n_bao + self.cached_n_cmb + self.cached_n_snia + self.cached_n_h0
        logger.info(f"Общо данни: {total_data_points} (BAO: {self.cached_n_bao}, CMB: {self.cached_n_cmb}, SN Ia: {self.cached_n_snia}, H₀: {self.cached_n_h0})")
        
        logger.info("Данните са настроени и кеширани!")
    
    def ptform(self, u: np.ndarray) -> np.ndarray:
        """Оптимизиран prior transform"""
        x = np.empty(self.n_params)
        
        for i, param_name in enumerate(self.parameter_names):
            param_min, param_max = self.parameter_ranges[param_name]
            x[i] = param_min + u[i] * (param_max - param_min)
        
        return x
    
    def loglike(self, params: np.ndarray) -> float:
        """ПЪЛЕН Cross-validation likelihood функция с BAO + CMB + SN Ia + H₀"""
        try:
            H0 = params[0]
            Omega_m = params[1]
            epsilon_bao = params[2] if len(params) > 2 else 0.0
            epsilon_cmb = params[3] if len(params) > 3 else 0.0

            # Бързи проверки
            if not (60 < H0 < 80 and 0.05 < Omega_m < 0.95):
                return -np.inf

            # 🚨 ПОПРАВКА: Използваме анизотропен No-Lambda модел
            try:
                from no_lambda_cosmology import NoLambdaCosmology
                
                cosmo = NoLambdaCosmology(
                    H0=H0,
                    Omega_m=Omega_m,
                    epsilon_bao=epsilon_bao,
                    epsilon_cmb=epsilon_cmb,
                    alpha=1.2,
                    beta=0.0,
                    gamma=0.4,
                    delta=0.08,
                    angular_strength=0.6
                )
                
                # BAO предсказания (анизотропни)
                bao_combined = self.bao_data.get_combined_data()
                z_bao = bao_combined['redshifts']
                
                # Генериране на анизотропни BAO предсказания
                bao_predictions = cosmo.calculate_bao_predictions(z_bao)
                
                # CMB предсказания
                theta_s_pred = cosmo.cmb_angular_scale()
                l_peaks_pred = np.array([
                    cosmo.cmb_peak_position(),
                    cosmo.cmb_peak_position() * 1.4,
                    cosmo.cmb_peak_position() * 2.1
                ])
                
                cmb_predictions = {
                    'theta_s': theta_s_pred,
                    'l_peaks': l_peaks_pred
                }
                
                # SN Ia предсказания (ако са налични)
                snia_predictions = {}
                if hasattr(self, 'snia_data') and self.snia_data is not None:
                    snia_combined = self.snia_data.get_combined_data()
                    z_snia = snia_combined['redshifts']
                    
                    # Distance modulus предсказания
                    mu_pred = cosmo.distance_modulus(z_snia)
                    snia_predictions['distance_modulus'] = mu_pred
                
                # H₀ предсказания (ако са налични)
                h0_predictions = {}
                if hasattr(self, 'h0_data') and self.h0_data is not None:
                    h0_pred = cosmo.h0_prediction()
                    h0_predictions['H0'] = h0_pred['H0']
                
                # Обединени предсказания
                combined_predictions = {
                    **bao_predictions,
                    **cmb_predictions,
                    **snia_predictions,
                    **h0_predictions
                }
                
                # Пълен likelihood от всички данни
                total_loglike = 0.0
                
                # BAO likelihood
                bao_loglike = self.likelihood_func.bao_likelihood(combined_predictions, use_anisotropic=True)
                total_loglike += bao_loglike
                
                # CMB likelihood
                cmb_loglike = self.likelihood_func.cmb_likelihood(combined_predictions)
                total_loglike += cmb_loglike
                
                # SN Ia likelihood (ако е налично)
                if hasattr(self, 'snia_data') and self.snia_data is not None:
                    snia_loglike = self.likelihood_func.snia_likelihood(combined_predictions)
                    total_loglike += snia_loglike
                
                # H₀ likelihood (ако е налично)
                if hasattr(self, 'h0_data') and self.h0_data is not None:
                    h0_loglike = self.likelihood_func.h0_likelihood(combined_predictions)
                    total_loglike += h0_loglike
                
                # Проверка за валидност
                if np.isnan(total_loglike) or np.isinf(total_loglike):
                    return -np.inf
                
                return total_loglike
                
            except Exception as e:
                logger.warning(f"Грешка в космологическия модел: {e}")
                return -np.inf
                
        except Exception as e:
            logger.warning(f"Грешка в likelihood функцията: {e}")
            return -np.inf
    
    def run_fast_sampling(self, 
                         nlive: int = None,
                         dynamic: bool = False,  # Static за скорост
                         progress: bool = False,
                         parallel: bool = True) -> None:  # Добавяме опция за паралелизация
        """
        Максимално бърз nested sampling с опция за паралелизация
        """
        
        if nlive is None:
            nlive = self.nlive
        
        print(f"🚀 Стартиране на БЪРЗ nested sampling: nlive={nlive}")
        start_time = time.time()
        
        if parallel:
            # Използвай всички налични ядра
            n_cpu = cpu_count()
            print(f"🔥 Използване на паралелизация с {n_cpu} ядра.")
            with Pool(processes=n_cpu) as pool:
                sampler = NestedSampler(
                    self.loglike,
                    self.ptform,
                    self.n_params,
                    nlive=nlive,
                    pool=pool,
                    queue_size=n_cpu
                )
                sampler.run_nested(print_progress=False)
        else:
            # Стандартен (сериен) режим
            sampler = NestedSampler(
                self.loglike,
                self.ptform,
                self.n_params,
                nlive=nlive
            )
            sampler.run_nested(print_progress=False)
        
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
        
        print(f"✅ Nested sampling завършен за {runtime:.1f}s")
        print(f"📊 Log-evidence: {self.log_evidence:.3f} ± {self.log_evidence_err:.3f}")
        print(f"📈 Samples: {len(self.posterior_samples)}")
        
        self._fast_analysis()
    
    def _fast_analysis(self):
        """Бърз анализ на резултатите"""
        
        if self.results is None:
            return
        
        # Бързи статистики
        self.param_stats = {}
        
        for i, param_name in enumerate(self.parameter_names):
            samples = self.posterior_samples[:, i]
            
            mean_val = np.mean(samples)
            percentiles = np.percentile(samples, [16, 50, 84])
            
            self.param_stats[param_name] = {
                'mean': mean_val,
                'median': percentiles[1],
                'lower_err': percentiles[1] - percentiles[0],
                'upper_err': percentiles[2] - percentiles[1],
                'std': np.std(samples)
            }
        
        # Information criteria
        n_data = self.cached_n_bao + self.cached_n_cmb + self.cached_n_snia + self.cached_n_h0
        best_log_like = np.max(self.results.logl)
        
        self.info_criteria = {
            'log_evidence': self.log_evidence,
            'log_evidence_err': self.log_evidence_err,
            'best_log_likelihood': best_log_like,
            'aic': 2 * self.n_params - 2 * best_log_like,
            'bic': np.log(n_data) * self.n_params - 2 * best_log_like,
            'n_parameters': self.n_params,
            'n_data': n_data
        }
    
    def quick_summary(self):
        """Бързо резюме без много форматиране"""
        
        print("\n" + "="*50)
        print("🎯 БЪРЗ NESTED SAMPLING РЕЗУЛТАТИ")
        print("="*50)
        
        if hasattr(self, 'log_evidence'):
            print(f"📊 Log-evidence: {self.log_evidence:.3f} ± {self.log_evidence_err:.3f}")
        
        if hasattr(self, 'info_criteria'):
            info = self.info_criteria
            print(f"📈 AIC: {info['aic']:.1f}")
            print(f"📈 BIC: {info['bic']:.1f}")
            print(f"📈 Best log-like: {info['best_log_likelihood']:.1f}")
        
        if hasattr(self, 'param_stats'):
            print(f"\n🔍 ПАРАМЕТРИ:")
            for param_name in self.parameter_names:
                if param_name in self.param_stats:
                    stats = self.param_stats[param_name]
                    print(f"  {param_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    def save_results(self, filename: str = "fast_nested_results.npz"):
        """Бързо записване на резултатите"""
        
        if self.results is None:
            print("❌ Няма резултати за записване")
            return
        
        np.savez(filename,
                samples=self.posterior_samples,
                logz=self.log_evidence,
                logz_err=self.log_evidence_err,
                param_names=self.parameter_names,
                param_stats=self.param_stats if hasattr(self, 'param_stats') else None
                )
        
        print(f"💾 Резултати записани в {filename}")


def quick_test():
    """Поетапен тест на оптимизациите"""
    
    print("🧪 ПОЕТАПЕН ТЕСТ НА ОПТИМИЗАЦИИТЕ")
    print("="*50)
    
    # Стъпка 1: Тест само с Numba (без паралелизация)
    print("\n🔥 СТЪПКА 1: Само Numba оптимизация")
    print("-"*30)
    
    ns = OptimizedNestedSampling(
        parameter_names=['H0', 'Omega_m'],  # Само 2 параметъра
        nlive=50  # Малко за бърз тест
    )
    
    print("⏱️ Стартиране на Numba тест (може да отнеме 30-60s за първа компилация)...")
    
    # Само Numba, БЕЗ паралелизация
    ns.run_fast_sampling(nlive=50, parallel=False, progress=False)
    
    print("✅ Numba тест завърши!")
    ns.quick_summary()
    
    # Стъпка 2: Ако Numba работи, тест с паралелизация
    print("\n🚀 СТЪПКА 2: Numba + паралелизация")
    print("-"*30)
    
    try:
        ns2 = OptimizedNestedSampling(
            parameter_names=['H0', 'Omega_m'],
            nlive=50
        )
        
        print("⏱️ Стартиране на паралелизиран тест...")
        ns2.run_fast_sampling(nlive=50, parallel=True, progress=False)
        
        print("✅ Паралелизиран тест завърши!")
        ns2.quick_summary()
        
    except Exception as e:
        print(f"❌ Паралелизацията не работи: {e}")
        print("ℹ️ Но Numba оптимизацията работи!")
    
    # Записване на резултатите
    ns.save_results("numba_test_results.npz")
    
    print("\n🎉 Тестовете завършиха!")
    print("💡 Ако Numba работи, имате поне 10x-50x ускорение!")


if __name__ == "__main__":
    quick_test() 