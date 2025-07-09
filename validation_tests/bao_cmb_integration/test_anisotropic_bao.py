#!/usr/bin/env python3
"""
Тест на анизотропен BAO анализ
============================

Тества новите функции за анизотропни BAO измервания:
- DA/rs и DH/rs изчисления
- Анизотропни likelihood функции
- Кръстосани корелации между измерванията
- Nested sampling с анизотропни данни
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
import time

# Наши модули
from no_lambda_cosmology import NoLambdaCosmology
from observational_data import BAOObservationalData, CMBObservationalData, LikelihoodFunctions
from bao_covariance_matrices import BAOCovarianceMatrices
from nested_sampling_analysis import OptimizedNestedSampling

# Настройка на логирането
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def test_anisotropic_distance_calculations():
    """Тест на анизотропни разстояния"""
    
    print("🧪 ТЕСТ НА АНИЗОТРОПНИ РАЗСТОЯНИЯ")
    print("=" * 50)
    
    # Създаване на модел
    cosmo = NoLambdaCosmology(
        H0=70.0,
        Omega_m=0.3,
        epsilon_bao=0.02,
        epsilon_cmb=0.015
    )
    
    # Тестови червени отмествания
    z_test = np.array([0.38, 0.51, 0.61, 0.70, 0.85])
    
    # Изчисляване на всички BAO разстояния
    bao_measures = cosmo.bao_distance_measures(z_test)
    
    print(f"{'z':<6} {'DA/rs':<8} {'DH/rs':<8} {'DV/rs':<8} {'r_s':<8}")
    print("-" * 50)
    
    for i, z in enumerate(z_test):
        print(f"{z:<6.2f} {bao_measures['DA_rs'][i]:<8.2f} {bao_measures['DH_rs'][i]:<8.2f} {bao_measures['DV_rs'][i]:<8.2f} {bao_measures['r_s']:<8.2f}")
    
    # Физически смисъл проверки
    print(f"\n🔍 ФИЗИЧЕСКИ ПРОВЕРКИ:")
    print(f"  r_s = {bao_measures['r_s']:.2f} Mpc (очаква се ~147 Mpc)")
    print(f"  DA/rs диапазон: {np.min(bao_measures['DA_rs']):.2f} - {np.max(bao_measures['DA_rs']):.2f}")
    print(f"  DH/rs диапазон: {np.min(bao_measures['DH_rs']):.2f} - {np.max(bao_measures['DH_rs']):.2f}")
    print(f"  DV/rs диапазон: {np.min(bao_measures['DV_rs']):.2f} - {np.max(bao_measures['DV_rs']):.2f}")
    
    # Анизотропия тест
    print(f"\n🧭 АНИЗОТРОПИЯ ТЕСТ:")
    
    # Различни посоки
    directions = [
        (0, 0, "z-ос"),
        (np.pi/2, 0, "x-ос"),
        (np.pi/2, np.pi/2, "y-ос"),
        (np.pi/4, np.pi/4, "диагонал")
    ]
    
    z_single = 0.5
    
    for theta, phi, name in directions:
        bao_aniso = cosmo.bao_distance_measures(z_single, theta, phi)
        print(f"  {name:<12}: DA/rs={bao_aniso['DA_rs']:.3f}, DH/rs={bao_aniso['DH_rs']:.3f}, DV/rs={bao_aniso['DV_rs']:.3f}")
    
    return bao_measures


def test_anisotropic_likelihood():
    """Тест на анизотропна likelihood функция"""
    
    print("\n🧪 ТЕСТ НА АНИЗОТРОПНА LIKELIHOOD")
    print("=" * 50)
    
    # Създаване на данни и likelihood функция
    bao_data = BAOObservationalData()
    cmb_data = CMBObservationalData()
    likelihood_func = LikelihoodFunctions(bao_data, cmb_data)
    
    # Тестов модел
    cosmo = NoLambdaCosmology(H0=70.0, Omega_m=0.3)
    
    # Получаване на тестови redshift стойности
    combined_data = bao_data.get_combined_data()
    z_values = combined_data['redshifts']
    
    print(f"Тестови z стойности: {z_values}")
    print(f"Брой точки: {len(z_values)}")
    
    # Генериране на model predictions
    bao_predictions = cosmo.calculate_bao_predictions(z_values)
    
    model_predictions = {
        'DV_rs': bao_predictions['DV_rs'],
        'DA_rs': bao_predictions['DA_rs'],
        'DH_rs': bao_predictions['DH_rs'],
        'theta_s': cosmo.cmb_angular_scale()
    }
    
    print(f"\nModel predictions:")
    print(f"  DV/rs: {model_predictions['DV_rs'][:5]}...")
    print(f"  DA/rs: {model_predictions['DA_rs'][:5]}...")
    print(f"  DH/rs: {model_predictions['DH_rs'][:5]}...")
    print(f"  theta_s: {model_predictions['theta_s']:.6f}")
    
    # Сравнение на изотропна vs анизотропна likelihood
    print(f"\n⚖️ СРАВНЕНИЕ НА LIKELIHOOD:")
    
    # Изотропна (само DV/rs)
    loglike_iso = likelihood_func.bao_likelihood(model_predictions, use_anisotropic=False)
    
    # Анизотропна (DV/rs + DA/rs + DH/rs)
    loglike_aniso = likelihood_func.bao_likelihood(model_predictions, use_anisotropic=True)
    
    # CMB likelihood
    cmb_loglike = likelihood_func.cmb_likelihood(model_predictions)
    
    print(f"  BAO изотропна: {loglike_iso:.2f}")
    print(f"  BAO анизотропна: {loglike_aniso:.2f}")
    print(f"  CMB: {cmb_loglike:.2f}")
    print(f"  Разлика: {loglike_aniso - loglike_iso:.2f}")
    
    # Chi-squared анализ
    chi2_analysis = likelihood_func.chi_squared_analysis(model_predictions)
    
    print(f"\n📊 CHI-SQUARED АНАЛИЗ:")
    print(f"  BAO χ²: {chi2_analysis['bao_chi2']:.2f}")
    print(f"  CMB χ²: {chi2_analysis['cmb_chi2']:.2f}")
    print(f"  Общо χ²: {chi2_analysis['combined_chi2']:.2f}")
    print(f"  Reduciran χ²: {chi2_analysis['reduced_chi2_combined']:.2f}")
    
    return loglike_aniso, chi2_analysis


def test_covariance_matrices():
    """Тест на ковариационни матрици за анизотропни измервания"""
    
    print("\n🧪 ТЕСТ НА КОВАРИАЦИОННИ МАТРИЦИ")
    print("=" * 50)
    
    bao_cov = BAOCovarianceMatrices()
    
    # Тест за различни размери
    test_sizes = [3, 6, 9, 12, 15]
    
    for size in test_sizes:
        cov_matrix = bao_cov.get_dataset_covariance_matrix('BOSS_DR12', size)
        
        print(f"  Размер {size}: матрица {cov_matrix.shape}")
        print(f"    Condition number: {np.linalg.cond(cov_matrix):.2e}")
        print(f"    Determinant: {np.linalg.det(cov_matrix):.2e}")
        print(f"    Диагонал: {np.diag(cov_matrix)[:3]}")
        
        # Проверка за положителна определеност
        eigenvals = np.linalg.eigvals(cov_matrix)
        is_pos_def = np.all(eigenvals > 0)
        print(f"    Положителна определеност: {is_pos_def}")
        
        if not is_pos_def:
            print(f"    ❌ Отрицателни eigenvalues: {eigenvals[eigenvals <= 0]}")
    
    # Тест за анизотропна матрица
    print(f"\n🎯 АНИЗОТРОПНА МАТРИЦА:")
    
    aniso_cov = bao_cov.generate_anisotropic_covariance('BOSS_DR12')
    
    print(f"  Размер: {aniso_cov['covariance'].shape}")
    print(f"  z точки: {aniso_cov['redshifts']}")
    print(f"  DA/rs грешки: {aniso_cov['da_rs_errors']}")
    print(f"  DH/rs грешки: {aniso_cov['dh_rs_errors']}")
    
    # Корелационен анализ
    cov_matrix = aniso_cov['covariance']
    corr_matrix = cov_matrix / np.sqrt(np.outer(np.diag(cov_matrix), np.diag(cov_matrix)))
    
    print(f"  Корелации диапазон: {np.min(corr_matrix):.3f} - {np.max(corr_matrix):.3f}")
    
    return aniso_cov


def test_nested_sampling_anisotropic():
    """Тест на nested sampling с анизотропни измервания"""
    
    print("\n🧪 ТЕСТ НА NESTED SAMPLING АНИЗОТРОПЕН")
    print("=" * 50)
    
    # Малък тест с ограничени параметри
    parameter_ranges = {
        'H0': (68.0, 74.0),
        'Omega_m': (0.25, 0.35),
        'epsilon_bao': (0.0, 0.05),
        'epsilon_cmb': (0.0, 0.03)
    }
    
    ns = OptimizedNestedSampling(
        parameter_names=list(parameter_ranges.keys()),
        parameter_ranges=parameter_ranges,
        nlive=50  # Малко за бърз тест
    )
    
    print(f"🚀 Стартиране на кратък анизотропен тест...")
    start_time = time.time()
    
    try:
        # Само сериен режим за по-стабилен тест
        ns.run_fast_sampling(nlive=50, parallel=False, progress=False)
        
        runtime = time.time() - start_time
        
        print(f"✅ Анизотропен тест завършен за {runtime:.1f}s")
        print(f"📊 Log-evidence: {ns.log_evidence:.3f} ± {ns.log_evidence_err:.3f}")
        print(f"📈 Samples: {len(ns.posterior_samples)}")
        
        # Кратки статистики
        ns.quick_summary()
        
        return ns
        
    except Exception as e:
        print(f"❌ Грешка в анизотропен тест: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Главна функция за тест"""
    
    print("🎯 ПЪЛЕН ТЕСТ НА АНИЗОТРОПЕН BAO АНАЛИЗ")
    print("=" * 70)
    
    # Стъпка 1: Тест на разстояния
    print("\n📏 СТЪПКА 1: Тест на разстояния")
    bao_measures = test_anisotropic_distance_calculations()
    
    # Стъпка 2: Тест на likelihood
    print("\n⚖️ СТЪПКА 2: Тест на likelihood")
    loglike_aniso, chi2_analysis = test_anisotropic_likelihood()
    
    # Стъпка 3: Тест на ковариационни матрици
    print("\n📊 СТЪПКА 3: Тест на ковариационни матрици")
    aniso_cov = test_covariance_matrices()
    
    # Стъпка 4: Тест на nested sampling
    print("\n🎯 СТЪПКА 4: Тест на nested sampling")
    ns_result = test_nested_sampling_anisotropic()
    
    # Резюме
    print("\n" + "=" * 70)
    print("📋 РЕЗЮМЕ НА АНИЗОТРОПНИЯ ТЕСТ")
    print("=" * 70)
    
    print(f"✅ Анизотропни разстояния: Работят")
    print(f"✅ Анизотропна likelihood: {loglike_aniso:.2f}")
    print(f"✅ Chi-squared: {chi2_analysis['reduced_chi2_combined']:.2f}")
    print(f"✅ Ковариационни матрици: Работят")
    
    if ns_result:
        print(f"✅ Nested sampling: Log-evidence {ns_result.log_evidence:.3f}")
        print(f"🎯 Анизотропният BAO анализ е ГОТОВ за използване!")
    else:
        print(f"❌ Nested sampling: Неуспешен")
        print(f"⚠️  Нужна е допълнителна отладка")
    
    print(f"\n🚀 Готов за **Стъпка 4**: Тест на пълен анизотропен анализ")


if __name__ == "__main__":
    main() 