#!/usr/bin/env python3
"""
Тестване на въздействието на пълните ковариационни матрици
спрямо диагонални матрици в BAO анализа
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from observational_data import create_bao_data, BAOObservationalData
from no_lambda_cosmology import NoLambdaCosmology
from nested_sampling_analysis import OptimizedNestedSampling
import time

# Настройка на логинг
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_covariance_impact():
    """
    Сравнение на резултатите с диагонални и пълни ковариационни матрици
    """
    
    print("�� ТЕСТВАНЕ НА ВЪЗДЕЙСТВИЕТО НА КОВАРИАЦИОННИТЕ МАТРИЦИ")
    print("=" * 70)
    
    # Тест параметри
    H0 = 72.0
    Omega_m = 0.30
    epsilon_bao = 0.01
    epsilon_cmb = 0.01
    
    # Създаване на модел
    cosmo = NoLambdaCosmology(H0=H0, Omega_m=Omega_m, epsilon_bao=epsilon_bao, epsilon_cmb=epsilon_cmb)
    
    # Зареждане на данни
    try:
        z_bao, DV_rs_obs, DV_rs_err, covariance_matrix = create_bao_data()
        print(f"✅ Заредени BAO данни: {len(z_bao)} точки")
        
        if covariance_matrix is not None:
            print(f"✅ Пълна ковариационна матрица: {covariance_matrix.shape}")
            has_full_covariance = True
        else:
            print("⚠️  Няма пълна ковариационна матрица - използване на диагонална")
            has_full_covariance = False
            
    except Exception as e:
        print(f"❌ Грешка при зареждане на данни: {e}")
        return
    
    # Модел предсказания
    DV_rs_model = []
    C_KM_S = 299792.458  # km/s
    
    for z in z_bao:
        D_A = cosmo.angular_diameter_distance(z)
        H_z = cosmo.hubble_function(z)
        D_H = C_KM_S / H_z
        D_V = (z * D_A**2 * D_H)**(1/3.0)
        r_s = cosmo.sound_horizon_scale()
        
        DV_rs_model.append(D_V / r_s)
    
    DV_rs_model = np.array(DV_rs_model)
    residuals = DV_rs_obs - DV_rs_model
    
    print(f"\n📊 СТАТИСТИКИ НА RESIDUALS:")
    print(f"   Средно отклонение: {np.mean(residuals):.4f}")
    print(f"   Стандартно отклонение: {np.std(residuals):.4f}")
    print(f"   Мин/Макс: {np.min(residuals):.4f} / {np.max(residuals):.4f}")
    
    # Сравнение на χ² изчислания
    print(f"\n🔍 СРАВНЕНИЕ НА χ² ИЗЧИСЛЕНИЯ:")
    print("-" * 40)
    
    # Диагонална χ²
    chi2_diagonal = np.sum((residuals / DV_rs_err)**2)
    print(f"📈 Диагонална χ²: {chi2_diagonal:.2f}")
    print(f"📈 Reduced χ² (диагонална): {chi2_diagonal/len(residuals):.2f}")
    
    # Пълна ковариационна χ²
    if has_full_covariance:
        try:
            cov_inv = np.linalg.inv(covariance_matrix)
            chi2_full = residuals.T @ cov_inv @ residuals
            print(f"📈 Пълна ковариационна χ²: {chi2_full:.2f}")
            print(f"📈 Reduced χ² (пълна): {chi2_full/len(residuals):.2f}")
            
            # Сравнение
            improvement = (chi2_diagonal - chi2_full) / chi2_diagonal * 100
            print(f"📊 Подобрение: {improvement:.1f}%")
            
            if improvement > 0:
                print("✅ Пълната ковариационна матрица дава по-добро фитване")
            else:
                print("⚠️  Диагоналната матрица дава по-добро фитване")
                
        except Exception as e:
            print(f"❌ Грешка при инвертиране на матрицата: {e}")
    
    # Анализ на корелации
    if has_full_covariance:
        print(f"\n🔍 АНАЛИЗ НА КОРЕЛАЦИИТЕ:")
        print("-" * 30)
        
        # Изчисляване на корелационна матрица
        diagonal_cov = np.diag(DV_rs_err**2)
        correlation_matrix = covariance_matrix / np.sqrt(np.outer(np.diag(covariance_matrix), np.diag(covariance_matrix)))
        
        # Стaтистики на корелациите
        off_diagonal = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        print(f"📊 Максимална корелация: {np.max(off_diagonal):.3f}")
        print(f"📊 Минимална корелация: {np.min(off_diagonal):.3f}")
        print(f"📊 Средна корелация: {np.mean(off_diagonal):.3f}")
        
        # Процент на значими корелации
        significant_correlations = np.sum(np.abs(off_diagonal) > 0.1)
        total_correlations = len(off_diagonal)
        print(f"📊 Значими корелации (|r|>0.1): {significant_correlations}/{total_correlations} ({significant_correlations/total_correlations*100:.1f}%)")
    
    print(f"\n✅ Анализът завърши успешно!")

def test_nested_sampling_comparison():
    """
    Сравнение на nested sampling с и без пълни ковариационни матрици
    """
    
    print("\n🚀 СРАВНЕНИЕ НА NESTED SAMPLING ПРОИЗВОДИТЕЛНОСТ")
    print("=" * 60)
    
    # Параметри за бърз тест
    nlive_test = 50  # Малко за бърз тест
    
    # Тест с текущата система (автоматично детектира коварианционни матрици)
    print("\n📊 Тест с текущата система:")
    print("-" * 30)
    
    start_time = time.time()
    
    sampler = OptimizedNestedSampling(
        parameter_names=['H0', 'Omega_m', 'epsilon_bao', 'epsilon_cmb'],
        nlive=nlive_test
    )
    
    try:
        sampler.run_fast_sampling(nlive=nlive_test, parallel=False)
        
        runtime = time.time() - start_time
        
        print(f"⏱️  Runtime: {runtime:.1f}s")
        print(f"📊 Log-evidence: {sampler.log_evidence:.3f} ± {sampler.log_evidence_err:.3f}")
        
        if hasattr(sampler, 'use_full_bao_covariance'):
            if sampler.use_full_bao_covariance:
                print("✅ Използвани пълни ковариационни матрици")
            else:
                print("⚠️  Използвани диагонални ковариационни матрици")
                
        # Резултати
        sampler.quick_summary()
        
    except Exception as e:
        print(f"❌ Грешка при nested sampling: {e}")
    
    print("\n✅ Тест завърши!")

if __name__ == "__main__":
    test_covariance_impact()
    test_nested_sampling_comparison() 