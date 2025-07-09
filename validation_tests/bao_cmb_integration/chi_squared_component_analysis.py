#!/usr/bin/env python3
"""
Детайлен анализ на χ² по компоненти - BAO vs CMB
Цел: Идентифициране на основните източници на високия χ²
"""

import numpy as np
import matplotlib.pyplot as plt
from no_lambda_cosmology import NoLambdaCosmology
from observational_data import BAOObservationalData, CMBObservationalData

def detailed_chi_squared_analysis():
    """Детайлен анализ на χ² по компоненти"""
    
    # Най-добрите параметри от nested sampling
    best_params = {
        'H0': 73.23,
        'Omega_m': 0.3046,
        'epsilon_bao': 0.0437,
        'epsilon_cmb': 0.0233
    }
    
    print("🔍 ДЕТАЙЛЕН АНАЛИЗ НА χ² ПО КОМПОНЕНТИ")
    print("=" * 60)
    print(f"📊 Най-добри параметри от nested sampling:")
    for param, value in best_params.items():
        print(f"   {param}: {value:.4f}")
    print()
    
    # Инициализация на модела
    cosmo = NoLambdaCosmology(**best_params)
    
    # Зареждане на данни
    bao_data = BAOObservationalData()
    cmb_data = CMBObservationalData()
    
    # BAO анализ
    print("📊 BAO АНАЛИЗ:")
    print("-" * 30)
    
    bao_combined = bao_data.get_combined_data()
    z_bao = bao_combined['redshifts']
    DV_rs_obs = bao_combined['DV_rs']
    DV_rs_err = bao_combined['DV_rs_err']
    
    # Изчисляване на теоретичните стойности
    DV_rs_theory = []
    chi2_individual_bao = []
    residuals_bao = []
    
    C_KM_S = 299792.458  # km/s
    
    for i, z in enumerate(z_bao):
        # Теоретично предсказание
        D_A = cosmo.angular_diameter_distance(z)
        H_z = cosmo.hubble_function(z)
        D_H = C_KM_S / H_z
        D_V = (z * D_A**2 * D_H)**(1/3.0)
        r_s = cosmo.sound_horizon_scale()
        
        DV_rs_model = D_V / r_s
        DV_rs_theory.append(DV_rs_model)
        
        # Residual и χ²
        residual = DV_rs_obs[i] - DV_rs_model
        chi2_point = (residual / DV_rs_err[i])**2
        
        residuals_bao.append(residual)
        chi2_individual_bao.append(chi2_point)
        
        print(f"   z={z:.3f}: obs={DV_rs_obs[i]:.3f}, model={DV_rs_model:.3f}, "
              f"res={residual:.3f}, χ²={chi2_point:.2f}")
    
    chi2_bao_total = np.sum(chi2_individual_bao)
    n_bao = len(z_bao)
    
    print(f"\n📊 BAO χ² резултати:")
    print(f"   Общо χ²_BAO: {chi2_bao_total:.2f}")
    print(f"   Точки: {n_bao}")
    print(f"   Reduced χ²_BAO: {chi2_bao_total/n_bao:.2f}")
    print()
    
    # CMB анализ
    print("📊 CMB АНАЛИЗ:")
    print("-" * 30)
    
    acoustic_data = cmb_data.get_acoustic_scale()
    theta_s_obs = acoustic_data['theta_s']
    theta_s_err = acoustic_data['theta_s_err']
    
    # Теоретично предсказание
    theta_s_model = cosmo.cmb_angular_scale()
    residual_cmb = theta_s_obs - theta_s_model
    chi2_cmb = (residual_cmb / theta_s_err)**2
    
    print(f"   θ_s observed: {theta_s_obs:.6f} rad")
    print(f"   θ_s model: {theta_s_model:.6f} rad")
    print(f"   Residual: {residual_cmb:.6f} rad")
    print(f"   Relative error: {100*residual_cmb/theta_s_obs:.2f}%")
    print(f"   χ²_CMB: {chi2_cmb:.2f}")
    print()
    
    # Обща статистика
    chi2_total = chi2_bao_total + chi2_cmb
    n_total = n_bao + 1  # BAO точки + 1 CMB точка
    n_params = 4  # H0, Omega_m, epsilon_bao, epsilon_cmb
    dof = n_total - n_params
    
    print("📊 ОБЩА СТАТИСТИКА:")
    print("=" * 40)
    print(f"χ²_BAO: {chi2_bao_total:.2f} ({chi2_bao_total/chi2_total*100:.1f}%)")
    print(f"χ²_CMB: {chi2_cmb:.2f} ({chi2_cmb/chi2_total*100:.1f}%)")
    print(f"χ²_total: {chi2_total:.2f}")
    print(f"DOF: {dof}")
    print(f"Reduced χ²: {chi2_total/dof:.2f}")
    print()
    
    # Идентификация на проблемни точки
    print("🔍 ПРОБЛЕМНИ ТОЧКИ:")
    print("-" * 30)
    
    # BAO точки с високи χ²
    high_chi2_bao = [(i, z_bao[i], chi2_individual_bao[i]) 
                     for i in range(len(chi2_individual_bao)) 
                     if chi2_individual_bao[i] > 5.0]
    
    if high_chi2_bao:
        print("BAO точки с χ² > 5.0:")
        for i, z, chi2 in high_chi2_bao:
            print(f"   z={z:.3f}: χ²={chi2:.2f}")
    else:
        print("Няма BAO точки с χ² > 5.0")
    
    if chi2_cmb > 5.0:
        print(f"CMB точка: χ²={chi2_cmb:.2f} (високо!)")
    else:
        print(f"CMB точка: χ²={chi2_cmb:.2f} (приемливо)")
    
    print()
    
    # Визуализация
    create_chi2_breakdown_plot(z_bao, chi2_individual_bao, chi2_cmb)
    
    # Препоръки
    print("🎯 ПРЕПОРЪКИ:")
    print("-" * 30)
    
    if chi2_bao_total > chi2_cmb:
        print("1. BAO данните доминират в χ² - фокус върху:")
        print("   - Ковариационна матрица за BAO")
        print("   - Систематични грешки в BAO measurements")
        print("   - epsilon_bao параметър оптимизация")
    else:
        print("1. CMB данните доминират в χ² - фокус върху:")
        print("   - CMB модел подобрения")
        print("   - epsilon_cmb параметър оптимизация")
    
    print("2. Общи препоръки:")
    print("   - Имплементация на пълната ковариационна матрица")
    print("   - Добавяне на SN Ia данни за cross-validation")
    print("   - Тестване на z-dependent epsilon параметри")
    
    return {
        'chi2_bao': chi2_bao_total,
        'chi2_cmb': chi2_cmb,
        'chi2_total': chi2_total,
        'dof': dof,
        'reduced_chi2': chi2_total/dof,
        'n_bao': n_bao,
        'high_chi2_bao': high_chi2_bao
    }

def create_chi2_breakdown_plot(z_bao, chi2_individual_bao, chi2_cmb):
    """Създаване на визуализация на χ² разпределението"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # BAO χ² по redshift
    ax1.bar(range(len(z_bao)), chi2_individual_bao, alpha=0.7, color='blue')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='χ² = 1 (идеално)')
    ax1.axhline(y=4.0, color='orange', linestyle='--', alpha=0.7, label='χ² = 4 (гранично)')
    ax1.set_xlabel('BAO точка index')
    ax1.set_ylabel('χ²')
    ax1.set_title('BAO χ² по точки')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Съотношение BAO vs CMB
    labels = ['BAO', 'CMB']
    sizes = [np.sum(chi2_individual_bao), chi2_cmb]
    colors = ['blue', 'red']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Разпределение на χ² между BAO и CMB')
    
    plt.tight_layout()
    plt.savefig('chi2_component_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Графика записана: chi2_component_breakdown.png")

if __name__ == "__main__":
    results = detailed_chi_squared_analysis()
    print("\n" + "="*60)
    print("✅ Анализът завършен!")
    print("📊 Следващи стъпки: Ковариационна матрица и SN Ia данни") 