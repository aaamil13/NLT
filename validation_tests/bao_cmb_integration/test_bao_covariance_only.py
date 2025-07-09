#!/usr/bin/env python3
"""
Фокусиран тест на BAO ковариационните матрици
"""

import numpy as np
from observational_data import create_bao_data
from bao_covariance_matrices import BAOCovarianceMatrices
from no_lambda_cosmology import NoLambdaCosmology

def test_bao_covariance_matrices():
    """
    Тест на BAO ковариационните матрици
    """
    
    print("🔍 ТЕСТ НА BAO КОВАРИАЦИОННИТЕ МАТРИЦИ")
    print("=" * 50)
    
    # Тест на BAOCovarianceMatrices класа
    print("\n📊 Тест на BAOCovarianceMatrices:")
    print("-" * 40)
    
    bao_cov = BAOCovarianceMatrices()
    
    # Проверка на наличните данни
    print(f"✅ Инициализиран BAOCovarianceMatrices")
    
    # Тест на create_bao_data
    print("\n📊 Тест на create_bao_data:")
    print("-" * 40)
    
    try:
        z_bao, DV_rs_obs, DV_rs_err, covariance_matrix = create_bao_data()
        
        print(f"✅ Заредени BAO данни:")
        print(f"   - Брой точки: {len(z_bao)}")
        print(f"   - Redshift range: {np.min(z_bao):.3f} - {np.max(z_bao):.3f}")
        print(f"   - DV/rs range: {np.min(DV_rs_obs):.3f} - {np.max(DV_rs_obs):.3f}")
        print(f"   - Грешки range: {np.min(DV_rs_err):.4f} - {np.max(DV_rs_err):.4f}")
        
        if covariance_matrix is not None:
            print(f"✅ Пълна ковариационна матрица: {covariance_matrix.shape}")
            print(f"   - Determinant: {np.linalg.det(covariance_matrix):.2e}")
            print(f"   - Condition number: {np.linalg.cond(covariance_matrix):.2e}")
            
            # Анализ на корелациите
            correlation_matrix = covariance_matrix / np.sqrt(np.outer(np.diag(covariance_matrix), np.diag(covariance_matrix)))
            off_diagonal = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            
            print(f"   - Макс корелация: {np.max(off_diagonal):.3f}")
            print(f"   - Мин корелация: {np.min(off_diagonal):.3f}")
            print(f"   - Средна корелация: {np.mean(off_diagonal):.3f}")
            
            # Тест на χ² изчислението
            test_chi_squared_calculation(z_bao, DV_rs_obs, DV_rs_err, covariance_matrix)
            
        else:
            print("⚠️  Няма пълна ковариационна матрица - fallback към диагонална")
            
    except Exception as e:
        print(f"❌ Грешка при create_bao_data: {e}")
        import traceback
        traceback.print_exc()

def test_chi_squared_calculation(z_bao, DV_rs_obs, DV_rs_err, covariance_matrix):
    """
    Тест на χ² изчислението с различни методи
    """
    
    print("\n🔍 ТЕСТ НА χ² ИЗЧИСЛЕНИЕТО:")
    print("-" * 40)
    
    # Тест параметри
    H0 = 72.0
    Omega_m = 0.30
    
    # Създаване на модел
    cosmo = NoLambdaCosmology(H0=H0, Omega_m=Omega_m)
    
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
    
    # Диагонална χ²
    chi2_diagonal = np.sum((residuals / DV_rs_err)**2)
    
    # Пълна ковариационна χ²
    if covariance_matrix is not None:
        try:
            cov_inv = np.linalg.inv(covariance_matrix)
            chi2_full = residuals.T @ cov_inv @ residuals
            
            print(f"📈 Диагонална χ²: {chi2_diagonal:.2f}")
            print(f"📈 Пълна ковариационна χ²: {chi2_full:.2f}")
            print(f"📈 Ratio (diagonal/full): {chi2_diagonal/chi2_full:.3f}")
            
            improvement = (chi2_diagonal - chi2_full) / chi2_diagonal * 100
            print(f"📊 Подобрение: {improvement:.1f}%")
            
            if improvement > 0:
                print("✅ Пълната ковариационна матрица дава по-добро фитване")
            else:
                print("⚠️  Диагоналната матрица дава по-добро фитване")
                
        except Exception as e:
            print(f"❌ Грешка при инвертиране на матрицата: {e}")
    
    # Детайлна статистика
    print(f"\n📊 Residuals статистика:")
    print(f"   - Средно: {np.mean(residuals):.4f}")
    print(f"   - Стандартно отклонение: {np.std(residuals):.4f}")
    print(f"   - Максимално отклонение: {np.max(np.abs(residuals)):.4f}")
    print(f"   - Pull статистика: {np.sqrt(chi2_diagonal/len(residuals)):.2f}")
    
    # Проверка на отделни точки
    print(f"\n📊 Анализ по точки:")
    for i, (z, obs, model, err) in enumerate(zip(z_bao, DV_rs_obs, DV_rs_model, DV_rs_err)):
        residual = obs - model
        pull = residual / err
        print(f"   z={z:.3f}: obs={obs:.3f}, model={model:.3f}, pull={pull:.2f}")

def test_matrix_properties():
    """
    Тест на свойствата на ковариационните матрици
    """
    
    print("\n🔍 ТЕСТ НА СВОЙСТВАТА НА МАТРИЦИТЕ:")
    print("-" * 40)
    
    bao_cov = BAOCovarianceMatrices()
    
    # Проверка на различни survey данни
    surveys = ['boss_dr12', 'eboss_dr16', '6dfgs', 'wigglez']
    
    for survey in surveys:
        print(f"\n📊 Survey: {survey}")
        try:
            # Тук може да добавим тестове за конкретни survey данни
            # когато имплементираме get_survey_data методи
            pass
        except Exception as e:
            print(f"   ❌ Грешка: {e}")

if __name__ == "__main__":
    test_bao_covariance_matrices()
    test_matrix_properties() 