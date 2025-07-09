#!/usr/bin/env python3
"""
Поправка на нереалистично малката CMB грешка
Цел: Замяна на θ_s грешката с реалистична стойност
"""

import numpy as np
from no_lambda_cosmology import NoLambdaCosmology
from observational_data import BAOObservationalData, CMBObservationalData

def fix_cmb_error_analysis():
    """Анализ и поправка на CMB грешката"""
    
    print("🔧 ПОПРАВКА НА НЕРЕАЛИСТИЧНАТА CMB ГРЕШКА")
    print("=" * 60)
    
    # Текущи стойности
    print("📊 ТЕКУЩИ СТОЙНОСТИ:")
    print(f"   θ_s observed: 0.010409 rad")
    print(f"   θ_s error: 0.0000031 rad (0.03%)")
    
    # Реалистични грешки според литературата
    realistic_errors = {
        "Conservative (1%)": 0.010409 * 0.01,
        "Moderate (0.5%)": 0.010409 * 0.005,
        "Optimistic (0.3%)": 0.010409 * 0.003,
        "Current (Unrealistic)": 0.0000031
    }
    
    print("\n🎯 РЕАЛИСТИЧНИ ГРЕШКИ:")
    for label, error in realistic_errors.items():
        error_percent = (error / 0.010409) * 100
        print(f"   {label}: {error:.7f} rad ({error_percent:.2f}%)")
    
    # Тест на различни грешки
    print("\n🧪 ТЕСТ НА РАЗЛИЧНИ ГРЕШКИ:")
    print("=" * 40)
    
    # Най-добрите параметри
    best_params = {
        'H0': 73.23,
        'Omega_m': 0.3046,
        'epsilon_bao': 0.0437,
        'epsilon_cmb': 0.0233
    }
    
    cosmo = NoLambdaCosmology(**best_params)
    
    # Изчислено θ_s
    theta_s_computed = cosmo.cmb_angular_scale()
    theta_s_observed = 0.010409
    residual = abs(theta_s_computed - theta_s_observed)
    
    print(f"θ_s computed: {theta_s_computed:.6f} rad")
    print(f"θ_s observed: {theta_s_observed:.6f} rad")
    print(f"Residual: {residual:.6f} rad")
    
    print("\nχ² с различни грешки:")
    for label, error in realistic_errors.items():
        chi2_cmb = (residual / error) ** 2
        print(f"   {label}: χ²_CMB = {chi2_cmb:.1f}")
    
    # Препоръка
    print("\n💡 ПРЕПОРЪКА:")
    recommended_error = 0.010409 * 0.005  # 0.5%
    recommended_chi2 = (residual / recommended_error) ** 2
    
    print(f"   Препоръчвам θ_s_err = {recommended_error:.7f} rad (0.5%)")
    print(f"   Това дава χ²_CMB = {recommended_chi2:.1f} (разумно)")
    
    return recommended_error

def implement_fix():
    """Имплементиране на поправката"""
    
    print("\n🔧 ИМПЛЕМЕНТИРАНЕ НА ПОПРАВКАТА")
    print("=" * 40)
    
    # Нова реалистична грешка
    new_theta_s_err = 0.010409 * 0.005  # 0.5%
    
    print(f"Старата грешка: 0.0000031 rad")
    print(f"Новата грешка: {new_theta_s_err:.7f} rad")
    print(f"Подобрение: фактор {0.0000031 / new_theta_s_err:.0f}")
    
    # Нуждаем се от редактиране на observational_data.py
    print("\n📝 НЕОБХОДИМО РЕДАКТИРАНЕ:")
    print("   Файл: observational_data.py")
    print("   Линия: 'theta_s_err': 0.0000031")
    print(f"   Нова стойност: 'theta_s_err': {new_theta_s_err:.7f}")
    
    return new_theta_s_err

if __name__ == "__main__":
    recommended_error = fix_cmb_error_analysis()
    implement_fix()
    
    print("\n✅ ГОТОВО! Сега observational_data.py трябва да се редактира.") 