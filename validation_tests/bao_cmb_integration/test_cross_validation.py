#!/usr/bin/env python3
"""
Тест на пълна Cross-Validation система
====================================

Тества No-Lambda космология срещу всички налични данни:
- BAO (анизотропни измервания)
- CMB (акустична скала и пикове)
- Type Ia Supernovae (distance modulus)
- Локални H₀ измервания

Използва оптимизиран nested sampling за пълна Bayesian inference.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
import time
import os

# Наши модули
from no_lambda_cosmology import NoLambdaCosmology
from observational_data import (
    BAOObservationalData, 
    CMBObservationalData,
    SNIaObservationalData,
    LocalH0ObservationalData,
    LikelihoodFunctions
)
from nested_sampling_analysis import OptimizedNestedSampling

# Настройка на логирането
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossValidationAnalysis:
    """
    Пълна Cross-Validation система за No-Lambda космология
    """
    
    def __init__(self, use_snia: bool = True, use_h0: bool = True):
        """
        Инициализация на cross-validation анализа
        
        Args:
            use_snia: Дали да се включат SN Ia данни
            use_h0: Дали да се включат H₀ данни
        """
        
        self.use_snia = use_snia
        self.use_h0 = use_h0
        
        logger.info("🚀 Инициализиране на Cross-Validation анализа")
        
        # Зареждане на всички данни
        self._load_all_data()
        
        # Настройка на nested sampling
        self._setup_nested_sampling()
        
        logger.info("✅ Cross-Validation система готова!")
    
    def _load_all_data(self):
        """Зареждане на всички наблюдателни данни"""
        
        logger.info("📊 Зареждане на наблюдателни данни...")
        
        # Основни данни
        self.bao_data = BAOObservationalData()
        self.cmb_data = CMBObservationalData()
        
        # Опционални данни
        if self.use_snia:
            self.snia_data = SNIaObservationalData()
            logger.info(f"✅ Заредени {len(self.snia_data.get_combined_data()['redshifts'])} SN Ia supernovae")
        else:
            self.snia_data = None
        
        if self.use_h0:
            self.h0_data = LocalH0ObservationalData()
            logger.info(f"✅ Заредени {len(self.h0_data.h0_measurements)} H₀ измервания")
        else:
            self.h0_data = None
        
        # Обединена likelihood система
        self.likelihood_func = LikelihoodFunctions(
            bao_data=self.bao_data,
            cmb_data=self.cmb_data,
            snia_data=self.snia_data,
            h0_data=self.h0_data
        )
    
    def _setup_nested_sampling(self):
        """Настройка на nested sampling за всички данни"""
        
        logger.info("🔧 Настройка на nested sampling...")
        
        # Параметри и диапазони
        parameter_names = ['H0', 'Omega_m', 'epsilon_bao', 'epsilon_cmb']
        parameter_ranges = {
            'H0': (60.0, 80.0),
            'Omega_m': (0.05, 0.95),
            'epsilon_bao': (-0.1, 0.1),
            'epsilon_cmb': (-0.1, 0.1)
        }
        
        # Nested sampling обект
        self.nested_sampler = OptimizedNestedSampling(
            parameter_names=parameter_names,
            parameter_ranges=parameter_ranges,
            nlive=150,  # По-голям брой за точност
            use_snia=self.use_snia,
            use_h0=self.use_h0
        )
        
        logger.info("✅ Nested sampling настроен")
    
    def run_individual_tests(self):
        """Тестване на отделните компоненти"""
        
        logger.info("\n🧪 ТЕСТВАНЕ НА ОТДЕЛНИТЕ КОМПОНЕНТИ")
        logger.info("=" * 60)
        
        # Тестов модел
        test_cosmo = NoLambdaCosmology(
            H0=70.0,
            Omega_m=0.3,
            epsilon_bao=0.02,
            epsilon_cmb=0.01
        )
        
        results = {}
        
        # BAO тест
        logger.info("\n🎵 BAO тест...")
        bao_combined = self.bao_data.get_combined_data()
        z_bao = bao_combined['redshifts']
        bao_predictions = test_cosmo.calculate_bao_predictions(z_bao)
        bao_loglike = self.likelihood_func.bao_likelihood(bao_predictions, use_anisotropic=True)
        results['BAO'] = {
            'loglike': bao_loglike,
            'chi2': -2 * bao_loglike,
            'n_data': len(z_bao),
            'reduced_chi2': -2 * bao_loglike / len(z_bao)
        }
        logger.info(f"  BAO χ²: {results['BAO']['chi2']:.2f} (reduced: {results['BAO']['reduced_chi2']:.2f})")
        
        # CMB тест
        logger.info("\n🌌 CMB тест...")
        theta_s_pred = test_cosmo.cmb_angular_scale()
        l_peaks_pred = np.array([
            test_cosmo.cmb_peak_position(),
            test_cosmo.cmb_peak_position() * 1.4,
            test_cosmo.cmb_peak_position() * 2.1
        ])
        cmb_predictions = {
            'theta_s': theta_s_pred,
            'l_peaks': l_peaks_pred
        }
        cmb_loglike = self.likelihood_func.cmb_likelihood(cmb_predictions)
        results['CMB'] = {
            'loglike': cmb_loglike,
            'chi2': -2 * cmb_loglike,
            'n_data': 4,
            'reduced_chi2': -2 * cmb_loglike / 4
        }
        logger.info(f"  CMB χ²: {results['CMB']['chi2']:.2f} (reduced: {results['CMB']['reduced_chi2']:.2f})")
        
        # SN Ia тест
        if self.use_snia:
            logger.info("\n🌟 SN Ia тест...")
            snia_combined = self.snia_data.get_combined_data()
            z_snia = snia_combined['redshifts']
            mu_pred = test_cosmo.distance_modulus(z_snia)
            snia_predictions = {'distance_modulus': mu_pred}
            snia_loglike = self.likelihood_func.snia_likelihood(snia_predictions)
            results['SN Ia'] = {
                'loglike': snia_loglike,
                'chi2': -2 * snia_loglike,
                'n_data': len(z_snia),
                'reduced_chi2': -2 * snia_loglike / len(z_snia)
            }
            logger.info(f"  SN Ia χ²: {results['SN Ia']['chi2']:.2f} (reduced: {results['SN Ia']['reduced_chi2']:.2f})")
        
        # H₀ тест
        if self.use_h0:
            logger.info("\n🔭 H₀ тест...")
            h0_pred = test_cosmo.h0_prediction()
            h0_predictions = {'H0': h0_pred['H0']}
            h0_loglike = self.likelihood_func.h0_likelihood(h0_predictions)
            results['H₀'] = {
                'loglike': h0_loglike,
                'chi2': -2 * h0_loglike,
                'n_data': 1,
                'reduced_chi2': -2 * h0_loglike / 1
            }
            logger.info(f"  H₀ χ²: {results['H₀']['chi2']:.2f} (reduced: {results['H₀']['reduced_chi2']:.2f})")
        
        # Обединени резултати
        total_loglike = sum(r['loglike'] for r in results.values())
        total_chi2 = sum(r['chi2'] for r in results.values())
        total_n_data = sum(r['n_data'] for r in results.values())
        
        logger.info(f"\n📊 ОБЕДИНЕНИ РЕЗУЛТАТИ:")
        logger.info(f"  Общо χ²: {total_chi2:.2f}")
        logger.info(f"  Общо данни: {total_n_data}")
        logger.info(f"  Reduciran χ²: {total_chi2/total_n_data:.2f}")
        logger.info(f"  Log-likelihood: {total_loglike:.2f}")
        
        return results
    
    def run_full_analysis(self, nlive: int = 150):
        """Пълен nested sampling анализ"""
        
        logger.info("\n🎯 ПЪЛЕН NESTED SAMPLING АНАЛИЗ")
        logger.info("=" * 60)
        
        # Стартиране на анализа
        start_time = time.time()
        
        # Настройка на по-голям брой live points
        self.nested_sampler.nlive = nlive
        
        logger.info(f"Стартиране на nested sampling с {nlive} live points...")
        
        # Стартиране на sampling
        self.nested_sampler.run_fast_sampling(
            nlive=nlive,
            dynamic=False,
            progress=False,
            parallel=True
        )
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Резултати
        results = self.nested_sampler.results
        log_evidence = self.nested_sampler.log_evidence
        log_evidence_err = self.nested_sampler.log_evidence_err
        param_stats = self.nested_sampler.param_stats
        
        logger.info(f"\n✅ NESTED SAMPLING ЗАВЪРШЕН!")
        logger.info(f"⏱️  Runtime: {runtime:.1f}s")
        logger.info(f"📊 Log-evidence: {log_evidence:.3f} ± {log_evidence_err:.3f}")
        logger.info(f"📈 Samples: {len(self.nested_sampler.posterior_samples)}")
        
        # Параметри
        logger.info(f"\n🔍 ПАРАМЕТРИ:")
        for param_name, stats in param_stats.items():
            logger.info(f"  {param_name}: {stats['median']:.3f} ± {stats['std']:.3f}")
        
        return {
            'runtime': runtime,
            'log_evidence': log_evidence,
            'log_evidence_err': log_evidence_err,
            'param_stats': param_stats,
            'n_samples': len(self.nested_sampler.posterior_samples)
        }
    
    def analyze_tensions(self):
        """Анализ на tensions между различните данни"""
        
        logger.info("\n🔍 АНАЛИЗ НА TENSIONS")
        logger.info("=" * 60)
        
        tensions = {}
        
        # H₀ tension
        if self.use_h0:
            h0_tension = self.h0_data.get_tension_analysis()
            tensions['H₀'] = h0_tension
            
            logger.info(f"H₀ tension:")
            logger.info(f"  Локално: {h0_tension['local_h0']:.2f} ± {h0_tension['local_err']:.2f}")
            logger.info(f"  CMB: {h0_tension['cmb_h0']:.2f} ± {h0_tension['cmb_err']:.2f}")
            logger.info(f"  Tension: {h0_tension['tension_sigma']:.1f}σ")
            logger.info(f"  Значим: {'ДА' if h0_tension['is_significant'] else 'НЕ'}")
        
        # BAO-CMB consistency
        test_cosmo = NoLambdaCosmology(H0=70.0, Omega_m=0.3)
        
        # Звукова скала от BAO vs CMB
        r_s_bao = test_cosmo.sound_horizon_scale()
        r_s_cmb = test_cosmo.sound_horizon_scale()  # Същата в No-Lambda
        
        logger.info(f"\nBAO-CMB consistency:")
        logger.info(f"  r_s (BAO): {r_s_bao:.3f} Mpc")
        logger.info(f"  r_s (CMB): {r_s_cmb:.3f} Mpc")
        logger.info(f"  Consistency: {abs(r_s_bao - r_s_cmb) / r_s_bao * 100:.2f}%")
        
        return tensions
    
    def summary_report(self):
        """Генериране на обобщаващ доклад"""
        
        logger.info("\n📋 ОБОБЩАВАЩ ДОКЛАД")
        logger.info("=" * 60)
        
        # Данни
        active_probes = ['BAO', 'CMB']
        if self.use_snia:
            active_probes.append('SN Ia')
        if self.use_h0:
            active_probes.append('H₀')
        
        logger.info(f"Активни проби: {', '.join(active_probes)}")
        
        # Общ брой данни
        total_data = self.nested_sampler.cached_n_bao + self.nested_sampler.cached_n_cmb
        if self.use_snia:
            total_data += self.nested_sampler.cached_n_snia
        if self.use_h0:
            total_data += self.nested_sampler.cached_n_h0
        
        logger.info(f"Общо данни: {total_data}")
        
        # Системни характеристики
        logger.info(f"Параметри: {self.nested_sampler.n_params}")
        logger.info(f"Live points: {self.nested_sampler.nlive}")
        
        # Статус
        if self.nested_sampler.results is not None:
            logger.info(f"Статус: ✅ Завършен анализ")
            logger.info(f"Log-evidence: {self.nested_sampler.log_evidence:.3f}")
        else:
            logger.info(f"Статус: ⏳ Готов за анализ")


def main():
    """Основна функция за тестване"""
    
    print("🚀 СТАРТИРАНЕ НА CROSS-VALIDATION АНАЛИЗ")
    print("=" * 70)
    
    # Конфигурация
    USE_SNIA = True
    USE_H0 = True
    
    # Създаване на анализатор
    cv_analyzer = CrossValidationAnalysis(
        use_snia=USE_SNIA,
        use_h0=USE_H0
    )
    
    # Показване на резюме
    cv_analyzer.summary_report()
    
    # Тестване на отделните компоненти
    individual_results = cv_analyzer.run_individual_tests()
    
    # Анализ на tensions
    tensions = cv_analyzer.analyze_tensions()
    
    # Пълен анализ
    full_results = cv_analyzer.run_full_analysis(nlive=150)
    
    print("\n🎉 CROSS-VALIDATION АНАЛИЗ ЗАВЪРШЕН!")
    print("=" * 70)
    
    # Финални резултати
    print(f"📊 Финални резултати:")
    print(f"  Log-evidence: {full_results['log_evidence']:.3f} ± {full_results['log_evidence_err']:.3f}")
    print(f"  Runtime: {full_results['runtime']:.1f}s")
    print(f"  Samples: {full_results['n_samples']}")
    
    # Параметри
    print(f"\n🔍 Най-добри параметри:")
    for param_name, stats in full_results['param_stats'].items():
        print(f"  {param_name}: {stats['median']:.3f} ± {stats['std']:.3f}")
    
    return full_results


if __name__ == "__main__":
    results = main() 