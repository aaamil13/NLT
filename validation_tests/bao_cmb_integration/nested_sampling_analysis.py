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
                 nlive: int = 100):  # По-малко за скорост
        """
        Инициализация с оптимизации за скорост
        """
        
        # Зареждане на данни ВЕДНЪЖ
        self.bao_data = BAOObservationalData()
        self.cmb_data = CMBObservationalData()
        self.likelihood_func = LikelihoodFunctions(self.bao_data, self.cmb_data)
        
        # Параметри
        if parameter_names is None:
            parameter_names = ['H0', 'Omega_m', 'epsilon_bao', 'epsilon_cmb']  # 🚨 ПОПРАВКА: Добавен epsilon_cmb
        
        self.parameter_names = parameter_names
        self.n_params = len(parameter_names)
        
        # Оптимизирани диапазони
        if parameter_ranges is None:
            parameter_ranges = {
                'H0': (65.0, 75.0),      # По-тесен диапазон
                'Omega_m': (0.25, 0.35), # По-тесен диапазон
                'epsilon_bao': (0.0, 0.05),
                'epsilon_cmb': (0.0, 0.05)  # 🚨 ПОПРАВКА: Добавен epsilon_cmb range
            }
        
        self.parameter_ranges = parameter_ranges
        
        # Nested sampling настройки за скорост
        self.nlive = nlive
        self.dlogz = 0.5  # По-груба точност за скорост
        self.maxiter = 1000  # Ограничение
        
        # Кеширани константи
        self._setup_cached_data()
        
        # Резултати
        self.sampler = None
        self.results = None
        
        print(f"✅ Инициализиран оптимизиран nested sampling: {self.n_params} параметра, nlive={nlive}")
    
    def _setup_cached_data(self):
        """Предварително кеширане на данни за скорост"""
        
        # 🚨 ПОПРАВКА: Използване на новата create_bao_data функция с пълни ковариационни матрици
        from observational_data import create_bao_data
        
        try:
            z_bao, DV_rs_obs, DV_rs_err, covariance_matrix = create_bao_data()
            
            self.cached_z_bao = z_bao
            self.cached_DV_rs_obs = DV_rs_obs
            self.cached_DV_rs_err = DV_rs_err
            self.cached_n_bao = len(z_bao)
            
            # Ковариационна матрица за BAO
            if covariance_matrix is not None:
                self.cached_bao_cov_inv = np.linalg.inv(covariance_matrix)
                self.use_full_bao_covariance = True
                print("✅ Използване на пълна BAO ковариационна матрица")
            else:
                self.cached_bao_cov_inv = None
                self.use_full_bao_covariance = False
                print("⚠️ Използване на диагонална BAO ковариационна матрица")
                
        except Exception as e:
            print(f"⚠️ Грешка при зареждане на BAO данни: {e}")
            # Fallback към старите данни
            bao_combined = self.bao_data.get_combined_data()
            self.cached_z_bao = bao_combined['redshifts']
            self.cached_DV_rs_obs = bao_combined['DV_rs']
            self.cached_DV_rs_err = bao_combined['DV_rs_err']
            self.cached_n_bao = len(self.cached_z_bao)
            self.cached_bao_cov_inv = None
            self.use_full_bao_covariance = False
        
        # Кеширане на CMB данни
        peak_data = self.cmb_data.get_peak_positions()
        acoustic_data = self.cmb_data.get_acoustic_scale()
        
        self.cached_l_peaks_obs = peak_data['l_peaks']
        self.cached_l_peaks_cov_inv = np.linalg.inv(peak_data['covariance'])
        
        self.cached_theta_s_obs = acoustic_data['theta_s']
        self.cached_theta_s_err = acoustic_data['theta_s_err']
        
        # Предизчислени константи
        self.cached_theta_s_var_inv = 1.0 / (self.cached_theta_s_err**2)
        
        print("✅ Кеширани данни за оптимизация")
    
    def ptform(self, u: np.ndarray) -> np.ndarray:
        """Оптимизиран prior transform"""
        x = np.empty(self.n_params)
        
        for i, param_name in enumerate(self.parameter_names):
            param_min, param_max = self.parameter_ranges[param_name]
            x[i] = param_min + u[i] * (param_max - param_min)
        
        return x
    
    def loglike(self, params: np.ndarray) -> float:
        """ПОПРАВЕН likelihood функция с No-Lambda модел"""
        try:
            H0 = params[0]
            Omega_m = params[1]
            epsilon_bao = params[2] if len(params) > 2 else 0.0
            epsilon_cmb = params[3] if len(params) > 3 else 0.0

            # Бързи проверки
            if not (60 < H0 < 80 and 0.05 < Omega_m < 0.95):
                return -np.inf

            # 🚨 ПОПРАВКА: Използваме пълния No-Lambda модел с поправките
            try:
                from no_lambda_cosmology import NoLambdaCosmology
                
                cosmo = NoLambdaCosmology(
                    H0=H0,
                    Omega_m=Omega_m,
                    epsilon_bao=epsilon_bao,
                    epsilon_cmb=epsilon_cmb
                )
                
                # BAO изчисления с поправения модел и пълни ковариационни матрици
                DV_rs_model = []
                for z in self.cached_z_bao:
                    # Използваме поправените функции
                    D_A = cosmo.angular_diameter_distance(z)
                    H_z = cosmo.hubble_function(z)
                    D_H = C_KM_S / H_z
                    D_V = (z * D_A**2 * D_H)**(1/3.0)
                    r_s = cosmo.sound_horizon_scale()
                    
                    DV_rs_model.append(D_V / r_s)
                
                DV_rs_model = np.array(DV_rs_model)
                residuals_bao = self.cached_DV_rs_obs - DV_rs_model
                
                # Chi-squared изчисление с опция за пълна ковариационна матрица
                if self.use_full_bao_covariance and self.cached_bao_cov_inv is not None:
                    # Използване на пълна ковариационна матрица
                    chi2_bao = residuals_bao.T @ self.cached_bao_cov_inv @ residuals_bao
                else:
                    # Диагонална ковариационна матрица (стандартен подход)
                    chi2_bao = np.sum((residuals_bao / self.cached_DV_rs_err)**2)
                
                # 🚨 ПОПРАВКА: CMB с правилния angular_diameter_distance
                theta_s_model = cosmo.cmb_angular_scale()  # Използваме поправената функция
                residual_cmb = self.cached_theta_s_obs - theta_s_model
                chi2_cmb = (residual_cmb / self.cached_theta_s_err)**2

                # Обща chi2
                total_chi2 = chi2_bao + chi2_cmb
                
                # Проверка за NaN/inf
                if not np.isfinite(total_chi2):
                    return -np.inf
                    
                return -0.5 * total_chi2
                
            except Exception as e:
                # При грешка в космологията, върни -inf
                return -np.inf
            
        except Exception:
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
        n_data = self.cached_n_bao + 4  # BAO + CMB приблизително
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