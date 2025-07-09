#!/usr/bin/env python3
"""
Детайлен анализ на компонентите на θ_s = r_s / D_A(z_drag)
Цел: Разбиране защо No-Λ моделът предсказва грешно CMB ъгловия мащаб
"""

import numpy as np
import matplotlib.pyplot as plt
from no_lambda_cosmology import NoLambdaCosmology
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

def analyze_theta_s_components():
    """Анализ на компонентите на θ_s"""
    
    print("🔍 АНАЛИЗ НА КОМПОНЕНТИТЕ НА θ_s")
    print("=" * 60)
    
    # Най-добрите параметри от nested sampling
    best_params = {
        'H0': 73.23,
        'Omega_m': 0.3046,
        'epsilon_bao': 0.0437,
        'epsilon_cmb': 0.0233
    }
    
    print(f"📊 Параметри: H₀={best_params['H0']:.2f}, Ωₘ={best_params['Omega_m']:.4f}")
    print(f"             ε_BAO={best_params['epsilon_bao']:.4f}, ε_CMB={best_params['epsilon_cmb']:.4f}")
    print()
    
    # === NO-LAMBDA МОДЕЛ ===
    print("�� NO-LAMBDA МОДЕЛ:")
    print("-" * 30)
    
    cosmo_nl = NoLambdaCosmology(
        H0=best_params['H0'],
        Omega_m=best_params['Omega_m'],
        epsilon_bao=best_params['epsilon_bao'],
        epsilon_cmb=best_params['epsilon_cmb']
    )
    
    # Ключови епохи
    z_drag_nl = cosmo_nl.z_drag
    z_star_nl = cosmo_nl.z_star
    
    print(f"   z_drag = {z_drag_nl:.1f}")
    print(f"   z_star = {z_star_nl:.1f}")
    
    # Звуков хоризонт
    r_s_nl = cosmo_nl.sound_horizon_scale(z_end=z_drag_nl)
    print(f"   r_s = {r_s_nl:.2f} Mpc")
    
    # Ъглово разстояние до drag epoch
    D_A_drag_nl = cosmo_nl.angular_diameter_distance(z_drag_nl)
    print(f"   D_A(z_drag) = {D_A_drag_nl:.2f} Mpc")
    
    # θ_s изчисление
    theta_s_nl = r_s_nl / D_A_drag_nl
    print(f"   θ_s = r_s / D_A = {theta_s_nl:.6f} rad")
    
    # === ΛCDM РЕФЕРЕНЦИЯ ===
    print("\n🌌 ΛCDM РЕФЕРЕНЦИЯ:")
    print("-" * 30)
    
    # Стандартни Planck 2018 параметри
    H0_lcdm = 67.4  # km/s/Mpc
    Omega_m_lcdm = 0.315
    Omega_b_lcdm = 0.049
    
    cosmo_lcdm = FlatLambdaCDM(
        H0=H0_lcdm * u.km / u.s / u.Mpc,
        Om0=Omega_m_lcdm,
        Ob0=Omega_b_lcdm
    )
    
    print(f"   H₀ = {H0_lcdm} km/s/Mpc")
    print(f"   Ωₘ = {Omega_m_lcdm}")
    print(f"   Ωᵦ = {Omega_b_lcdm}")
    
    # Приблизителен z_drag за ΛCDM
    z_drag_lcdm = 1060  # Стандартна стойност
    
    # Звуков хоризонт за ΛCDM (използвайки стандартна формула)
    # Приблизителна стойност от литературата
    r_s_lcdm = 147.0  # Mpc (Planck 2018)
    
    # Ъглово разстояние за ΛCDM
    D_A_drag_lcdm = cosmo_lcdm.angular_diameter_distance(z_drag_lcdm).value
    
    print(f"   z_drag ≈ {z_drag_lcdm}")
    print(f"   r_s ≈ {r_s_lcdm:.2f} Mpc")
    print(f"   D_A(z_drag) = {D_A_drag_lcdm:.2f} Mpc")
    
    # θ_s за ΛCDM
    theta_s_lcdm = r_s_lcdm / D_A_drag_lcdm
    print(f"   θ_s = r_s / D_A = {theta_s_lcdm:.6f} rad")
    
    # === ЕКСПЕРИМЕНТАЛНА СТОЙНОСТ ===
    print("\n🔬 ЕКСПЕРИМЕНТАЛНА СТОЙНОСТ:")
    print("-" * 30)
    theta_s_obs = 0.010409
    theta_s_err = 0.0000031
    print(f"   θ_s = {theta_s_obs:.6f} ± {theta_s_err:.7f} rad")
    
    # === СРАВНЕНИЕ ===
    print("\n📊 СРАВНЕНИЕ:")
    print("=" * 60)
    
    # Резидуали
    residual_nl = theta_s_nl - theta_s_obs
    residual_lcdm = theta_s_lcdm - theta_s_obs
    
    # Относителни грешки
    error_nl = 100 * residual_nl / theta_s_obs
    error_lcdm = 100 * residual_lcdm / theta_s_obs
    
    # Sigma отклонения
    sigma_nl = abs(residual_nl) / theta_s_err
    sigma_lcdm = abs(residual_lcdm) / theta_s_err
    
    print(f"{'-':<12} {'θ_s':<10} {'Резидуал':<12} {'Грешка %':<10} {'Sigma':<10}")
    print("-" * 60)
    print(f"{'-':<12} {theta_s_nl:<10.6f} {residual_nl:<12.6f} {error_nl:<10.2f} {sigma_nl:<10.1f}")
    print(f"{'-':<12} {theta_s_lcdm:<10.6f} {residual_lcdm:<12.6f} {error_lcdm:<10.2f} {sigma_lcdm:<10.1f}")
    print(f"{'-':<12} {theta_s_obs:<10.6f} {'0.000000':<12} {'0.00':<10} {'0.0':<10}")
    
    # === КОМПОНЕНТЕН АНАЛИЗ ===
    print("\n🔍 КОМПОНЕНТЕН АНАЛИЗ:")
    print("=" * 60)
    
    # r_s сравнение
    r_s_ratio = r_s_nl / r_s_lcdm
    print("Звуков хоризонт r_s:")
    print(f"   No-Λ: {r_s_nl:.2f} Mpc")
    print(f"   ΛCDM: {r_s_lcdm:.2f} Mpc")
    print(f"   Ratio: {r_s_ratio:.3f} ({(r_s_ratio-1)*100:+.1f}%)")
    
    # D_A сравнение
    D_A_ratio = D_A_drag_nl / D_A_drag_lcdm
    print(f"\nЪглово разстояние D_A(z_drag):")
    print(f"   No-Λ: {D_A_drag_nl:.2f} Mpc")
    print(f"   ΛCDM: {D_A_drag_lcdm:.2f} Mpc")
    print(f"   Ratio: {D_A_ratio:.3f} ({(D_A_ratio-1)*100:+.1f}%)")
    
    # θ_s сравнение
    theta_s_ratio = theta_s_nl / theta_s_lcdm
    print(f"\nЪглов мащаб θ_s = r_s / D_A:")
    print(f"   No-Λ: {theta_s_nl:.6f} rad")
    print(f"   ΛCDM: {theta_s_lcdm:.6f} rad")
    print(f"   Ratio: {theta_s_ratio:.3f} ({(theta_s_ratio-1)*100:+.1f}%)")
    
    # === ДИАГНОЗА ===
    print("\n🎯 ДИАГНОЗА:")
    print("=" * 60)
    
    if abs(error_nl) > 10:
        print(f"❌ No-Λ моделът има {error_nl:.1f}% грешка в θ_s ({sigma_nl:.0f}-sigma отклонение)")
        
        if abs(r_s_ratio - 1) > 0.1:
            print(f"⚠️  Звуковият хоризонт r_s е {(r_s_ratio-1)*100:+.1f}% различен от ΛCDM")
        
        if abs(D_A_ratio - 1) > 0.1:
            print(f"⚠️  Ъгловото разстояние D_A е {(D_A_ratio-1)*100:+.1f}% различно от ΛCDM")
        
        print("\n🔧 НЕОБХОДИМИ КОРЕКЦИИ:")
        if r_s_ratio < 0.9:
            print("   - Звуковият хоризонт е твърде малък")
        elif r_s_ratio > 1.1:
            print("   - Звуковият хоризонт е твърде голям")
        
        if D_A_ratio < 0.9:
            print("   - Ъгловото разстояние е твърде малко")
        elif D_A_ratio > 1.1:
            print("   - Ъгловото разстояние е твърде голямо")
        
        print("\n💡 ПРЕДЛОЖЕНИЯ:")
        print("   1. Преразгледайте калибрацията на z_drag")
        print("   2. Проверете формулата за sound horizon")
        print("   3. Анализирайте влиянието на epsilon_cmb")
        print("   4. Разгледайте физическите корекции в модела")
    
    else:
        print(f"✅ Моделът има приемлива грешка от {error_nl:.1f}%")
    
    return {
        'theta_s_nl': theta_s_nl,
        'theta_s_lcdm': theta_s_lcdm,
        'theta_s_obs': theta_s_obs,
        'r_s_nl': r_s_nl,
        'r_s_lcdm': r_s_lcdm,
        'D_A_nl': D_A_drag_nl,
        'D_A_lcdm': D_A_drag_lcdm,
        'error_nl': error_nl,
        'sigma_nl': sigma_nl
    }

if __name__ == "__main__":
    results = analyze_theta_s_components()
    
    print("\n" + "=" * 60)
    print("✅ АНАЛИЗ ЗАВЪРШЕН!")
    print(f"📊 No-Λ грешка: {results['error_nl']:.1f}% ({results['sigma_nl']:.0f}-sigma)")
    print("🎯 Следваща стъпка: Подобрение на физическия модел")
    print("=" * 60) 