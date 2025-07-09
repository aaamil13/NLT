#!/usr/bin/env python3
"""
DEBUG CMB Angular Scale
=======================

Специализиран дебъгване за CMB θ_s проблема:
- Наблюдавано: θ_s = 0.010409
- Теоретично: θ_s = 6.621363 (637x по-голямо!)
- Относителна грешка: -63,510.7%
"""

import numpy as np
from no_lambda_cosmology import NoLambdaCosmology
from observational_data import CMBObservationalData
from scipy.integrate import quad

def debug_cmb_angular_scale():
    """Детайлен дебъгване на CMB angular scale"""
    
    print("🔍 DEBUG CMB ANGULAR SCALE")
    print("=" * 50)
    
    # Best-fit параметри
    params = {
        'H0': 68.4557,
        'Omega_m': 0.2576,
        'epsilon_bao': 0.0492,
        'epsilon_cmb': 0.0225
    }
    
    print(f"📊 Best-fit параметри: {params}")
    
    # Създаване на модел
    cosmo = NoLambdaCosmology(
        H0=params['H0'],
        Omega_m=params['Omega_m'],
        epsilon_bao=params['epsilon_bao'],
        epsilon_cmb=params['epsilon_cmb']
    )
    
    print(f"\n📊 Космологични параметри:")
    print(f"  H₀ = {cosmo.H0:.2f} km/s/Mpc")
    print(f"  Ωₘ = {cosmo.Omega_m:.4f}")
    print(f"  Ωᵦ = {cosmo.Omega_b:.4f}")
    print(f"  Ωᵣ = {cosmo.Omega_r:.2e}")
    print(f"  Ωₖ = {cosmo.Omega_k:.4f} 🚨 ОГРОМНО!")
    print(f"  z_drag = {cosmo.z_drag:.1f}")
    
    # CMB данни
    cmb_data = CMBObservationalData()
    cmb_obs = cmb_data.get_acoustic_scale()
    
    print(f"\n📊 CMB данни:")
    print(f"  Наблюдавано θ_s = {cmb_obs['theta_s']:.6f} rad")
    print(f"  Грешка = {cmb_obs['theta_s_err']:.2e} rad")
    
    # Стъпка по стъпка изчисление
    print(f"\n🔍 СТЪПКА ПО СТЪПКА ИЗЧИСЛЕНИЕ")
    print("=" * 50)
    
    # 1. Sound horizon
    z_cmb = 1090.0
    r_s = cosmo.sound_horizon_scale(cosmo.z_drag)
    print(f"1. Sound horizon r_s = {r_s:.2f} Mpc")
    
    # 2. Angular diameter distance - детайлно
    print(f"2. Angular diameter distance D_A(z={z_cmb}):")
    
    # Comoving distance
    def integrand(z):
        H_z = cosmo.hubble_function(z)
        return (299792.458 / 1000) / H_z  # km/s -> Mpc
    
    comoving_dist, _ = quad(integrand, 0, z_cmb)
    print(f"   Comoving distance D_M = {comoving_dist:.1f} Mpc")
    
    # Open universe correction
    Omega_k = cosmo.Omega_k
    if abs(Omega_k) > 1e-6:
        sqrt_Ok = np.sqrt(abs(Omega_k))
        DH = (299792.458 / 1000) / cosmo.H0  # Hubble distance
        print(f"   √|Ωₖ| = {sqrt_Ok:.4f}")
        print(f"   Hubble distance D_H = {DH:.1f} Mpc")
        
        if Omega_k > 0:  # Open universe
            argument = sqrt_Ok * comoving_dist / DH
            print(f"   Sinh argument = {argument:.4f}")
            transverse_dist = DH / sqrt_Ok * np.sinh(argument)
            print(f"   Transverse distance D_T = {transverse_dist:.1f} Mpc")
        else:
            argument = sqrt_Ok * comoving_dist / DH
            transverse_dist = DH / sqrt_Ok * np.sin(argument)
            print(f"   Transverse distance D_T = {transverse_dist:.1f} Mpc")
    else:
        transverse_dist = comoving_dist
        print(f"   Transverse distance D_T = {transverse_dist:.1f} Mpc (плоска)")
    
    # Angular diameter distance
    D_A = transverse_dist / (1 + z_cmb)
    print(f"   Angular diameter distance D_A = {D_A:.1f} Mpc")
    
    # 3. Angular scale
    theta_s_theory = r_s / D_A
    print(f"3. Angular scale θ_s = r_s/D_A = {theta_s_theory:.6f} rad")
    
    # 4. Сравнение
    print(f"\n📊 СРАВНЕНИЕ:")
    print(f"  Наблюдавано: θ_s = {cmb_obs['theta_s']:.6f} rad")
    print(f"  Теоретично:  θ_s = {theta_s_theory:.6f} rad")
    print(f"  Отношение: theory/obs = {theta_s_theory/cmb_obs['theta_s']:.1f}")
    print(f"  Разлика: {abs(theta_s_theory - cmb_obs['theta_s']):.6f} rad")
    
    # 5. Диагноза
    print(f"\n🚨 ДИАГНОЗА:")
    if theta_s_theory > 100 * cmb_obs['theta_s']:
        print("  КРИТИЧНО: Теоретичната стойност е >100x по-голяма!")
        print("  Проблемът е най-вероятно в:")
        print("    - Open universe геометрията (Ωₖ = 0.6849)")
        print("    - Angular diameter distance изчислението")
        print("    - Sinh функцията за open universe")
    
    # 6. Референтна проверка
    print(f"\n🔍 РЕФЕРЕНТНА ПРОВЕРКА:")
    print("  Типични стойности за ΛCDM:")
    print("    - r_s ≈ 147 Mpc")
    print("    - D_A(1090) ≈ 14,000 Mpc")
    print("    - θ_s ≈ 0.0104 rad")
    
    print(f"  Нашите стойности:")
    print(f"    - r_s = {r_s:.1f} Mpc ({'✅' if 140 < r_s < 150 else '❌'})")
    print(f"    - D_A(1090) = {D_A:.1f} Mpc ({'✅' if 13000 < D_A < 15000 else '❌'})")
    print(f"    - θ_s = {theta_s_theory:.6f} rad ({'✅' if 0.009 < theta_s_theory < 0.012 else '❌'})")
    
    # 7. Предложения за решение
    print(f"\n💡 ПРЕДЛОЖЕНИЯ ЗА РЕШЕНИЕ:")
    print("  1. Ограничаване на Ωₖ до физически стойности (-0.1 < Ωₖ < 0.1)")
    print("  2. Използване на flat universe приближение")
    print("  3. Корекция на D_A изчислението за extreme curvature")
    print("  4. Използване на референтни стойности за D_A")
    
    return {
        'r_s': r_s,
        'D_A': D_A,
        'theta_s_theory': theta_s_theory,
        'theta_s_obs': cmb_obs['theta_s'],
        'ratio': theta_s_theory/cmb_obs['theta_s'],
        'Omega_k': Omega_k,
        'comoving_dist': comoving_dist,
        'transverse_dist': transverse_dist
    }

def test_flat_universe_approximation():
    """Тест с flat universe приближение"""
    
    print(f"\n🧪 ТЕСТ С FLAT UNIVERSE ПРИБЛИЖЕНИЕ")
    print("=" * 50)
    
    # Същите параметри, но с Ωₖ = 0
    params = {
        'H0': 68.4557,
        'Omega_m': 0.2576,
        'epsilon_bao': 0.0492,
        'epsilon_cmb': 0.0225
    }
    
    # Принудително flat universe
    cosmo = NoLambdaCosmology(
        H0=params['H0'],
        Omega_m=params['Omega_m'],
        epsilon_bao=params['epsilon_bao'],
        epsilon_cmb=params['epsilon_cmb']
    )
    
    # Temporarily override Omega_k
    cosmo.Omega_k = 0.0  # Flat universe
    cosmo.Omega_Lambda = 1.0 - cosmo.Omega_m - cosmo.Omega_r  # Compensate
    
    print(f"📊 Flat universe параметри:")
    print(f"  Ωₖ = {cosmo.Omega_k:.1f} (принудително)")
    print(f"  ΩΛ = {cosmo.Omega_Lambda:.4f} (компенсация)")
    
    # Изчисляване на θ_s
    z_cmb = 1090.0
    r_s = cosmo.sound_horizon_scale(cosmo.z_drag)
    D_A = cosmo.angular_diameter_distance(z_cmb)
    theta_s_flat = r_s / D_A
    
    print(f"  r_s = {r_s:.1f} Mpc")
    print(f"  D_A(1090) = {D_A:.1f} Mpc")
    print(f"  θ_s = {theta_s_flat:.6f} rad")
    
    # Сравнение
    cmb_data = CMBObservationalData()
    cmb_obs = cmb_data.get_acoustic_scale()
    
    print(f"\n📊 Flat universe резултати:")
    print(f"  Наблюдавано: θ_s = {cmb_obs['theta_s']:.6f} rad")
    print(f"  Flat theory: θ_s = {theta_s_flat:.6f} rad")
    print(f"  Отношение: theory/obs = {theta_s_flat/cmb_obs['theta_s']:.1f}")
    
    improvement = theta_s_flat < 0.1  # Реалистично подобрение
    print(f"  Резултат: {'✅ ПОДОБРЕНИЕ' if improvement else '❌ НЕ ПОМАГА'}")
    
    return theta_s_flat

if __name__ == "__main__":
    print("🚨 DEBUG CMB ANGULAR SCALE PROBLEM")
    print("🎯 Цел: Намиране на причината за θ_s проблема")
    print("=" * 60)
    
    # Основен дебъгване
    results = debug_cmb_angular_scale()
    
    # Тест с flat universe
    theta_s_flat = test_flat_universe_approximation()
    
    print(f"\n🎉 ДЕБЪГВАНЕТО ЗАВЪРШИ!")
    print(f"📋 Проверете резултатите за възможни решения") 