#!/usr/bin/env python3
"""
Поправка на angular_diameter_distance изчислението
Цел: Диагностициране и корекция на проблемите с единици и формулировка
"""

import numpy as np
from scipy import integrate
import logging
from no_lambda_cosmology import NoLambdaCosmology

# Физически константи
c = 299792458  # м/с

def debug_angular_diameter_calculation():
    """Детайлно дебъгване на D_A изчислението"""
    
    print("🔍 ДЕБЪГВАНЕ НА ANGULAR DIAMETER DISTANCE")
    print("=" * 60)
    
    # Тестове параметри
    H0 = 73.23
    Omega_m = 0.3046
    
    print(f"📊 Параметри: H₀={H0}, Ωₘ={Omega_m}")
    
    # Създаване на модел
    cosmo = NoLambdaCosmology(H0=H0, Omega_m=Omega_m)
    
    # Тестови z стойности
    z_test_values = [1076.8, 1090.0, 100.0, 1.0, 0.1]
    
    print("\n🧪 ТЕСТ НА РАЗЛИЧНИ Z СТОЙНОСТИ:")
    print("-" * 60)
    print(f"{'|z':<10} {'D_A (текуща)':<15} {'D_A (очаквана)':<15} {'Статус':<10}")
    print("-" * 60)
    
    for z in z_test_values:
        # Текуща стойност
        D_A_current = cosmo.angular_diameter_distance(z)
        
        # Очаквана стойност (приблизителна)
        if z > 1000:
            D_A_expected = 14000  # Mpc за високо z
        elif z > 10:
            D_A_expected = 1000 + z * 10  # Приблизително за средно z
        else:
            D_A_expected = z * 1000  # Приблизително за ниско z
        
        # Статус
        ratio = D_A_current / D_A_expected
        if 0.5 < ratio < 2.0:
            status = "✅ OK"
        else:
            status = "❌ ГРЕШКА"
        
        print(f"{z:<10.1f} {D_A_current:<15.2f} {D_A_expected:<15.2f} {status:<10}")
    
    # Детайлно дебъгване за z_drag
    print("\n🔍 ДЕТАЙЛНО ДЕБЪГВАНЕ ЗА z_drag:")
    print("-" * 60)
    
    z_drag = cosmo.z_drag
    print(f"z_drag = {z_drag:.1f}")
    
    # Проверка на интеграл функцията
    print("\n📊 ПРОВЕРКА НА ИНТЕГРАЛА:")
    
    def integrand_test(z_val):
        H_z = cosmo.hubble_function(z_val)  # км/с/Mpc
        return (c / 1000) / H_z  # km -> Mpc
    
    # Тест на интеграл
    try:
        comoving_distance, _ = integrate.quad(integrand_test, 0, z_drag, 
                                            epsabs=1e-10, epsrel=1e-8)
        print(f"Comoving distance: {comoving_distance:.2f} Mpc")
        
        # Корекция за кривина
        if abs(cosmo.Omega_k) > 1e-6:
            sqrt_Ok = np.sqrt(abs(cosmo.Omega_k))
            DH = (c / 1000) / cosmo.H0  # Hubble distance в Mpc
            
            print(f"Omega_k = {cosmo.Omega_k:.4f}")
            print(f"sqrt(|Omega_k|) = {sqrt_Ok:.4f}")
            print(f"DH = {DH:.2f} Mpc")
            
            if cosmo.Omega_k > 0:  # Отворена Вселена
                transverse_distance = DH / sqrt_Ok * np.sinh(sqrt_Ok * comoving_distance / DH)
                print(f"Отворена Вселена: transverse_distance = {transverse_distance:.2f} Mpc")
            else:  # Затворена Вселена
                transverse_distance = DH / sqrt_Ok * np.sin(sqrt_Ok * comoving_distance / DH)
                print(f"Затворена Вселена: transverse_distance = {transverse_distance:.2f} Mpc")
        else:
            transverse_distance = comoving_distance
            print(f"Плоска Вселена: transverse_distance = {transverse_distance:.2f} Mpc")
        
        # Ъглово разстояние
        D_A_computed = transverse_distance / (1 + z_drag)
        print(f"D_A = transverse_distance / (1 + z) = {D_A_computed:.2f} Mpc")
        
        # Сравнение с функцията
        D_A_function = cosmo.angular_diameter_distance(z_drag)
        print(f"D_A (функция) = {D_A_function:.2f} Mpc")
        
        print(f"Разлика: {abs(D_A_computed - D_A_function):.2f} Mpc")
        
    except Exception as e:
        print(f"Грешка в интеграла: {e}")
    
    return cosmo, z_drag

def create_corrected_angular_diameter_distance():
    """Създаване на поправена функция за angular diameter distance"""
    
    print("\n🔧 СЪЗДАВАНЕ НА ПОПРАВЕНА ФУНКЦИЯ")
    print("=" * 60)
    
    def corrected_angular_diameter_distance(cosmo, z, theta=0, phi=0):
        """Поправена функция за angular diameter distance"""
        z = np.asarray(z)
        
        def integrand(z_val):
            H_z = cosmo.hubble_function(z_val, theta, phi)  # км/с/Mpc
            return (c / 1000) / H_z  # Преобразуване в Mpc
        
        D_A = np.zeros_like(z)
        
        for i, z_val in enumerate(z.flat):
            if z_val > 0:
                try:
                    # Коморбидно разстояние
                    comoving_distance, _ = integrate.quad(integrand, 0, z_val,
                                                         epsabs=1e-10, epsrel=1e-8)
                    
                    # Корекция за кривина
                    if abs(cosmo.Omega_k) > 1e-6:
                        sqrt_Ok = np.sqrt(abs(cosmo.Omega_k))
                        DH = (c / 1000) / cosmo.H0  # Hubble distance в Mpc
                        
                        if cosmo.Omega_k > 0:  # Отворена Вселена
                            transverse_distance = DH / sqrt_Ok * np.sinh(sqrt_Ok * comoving_distance / DH)
                        else:  # Затворена Вселена
                            transverse_distance = DH / sqrt_Ok * np.sin(sqrt_Ok * comoving_distance / DH)
                    else:
                        transverse_distance = comoving_distance
                    
                    # Ъглово разстояние
                    D_A.flat[i] = transverse_distance / (1 + z_val)
                    
                except Exception as e:
                    print(f"Грешка при z={z_val}: {e}")
                    # Fallback към приближение
                    D_A.flat[i] = (c / 1000) * z_val / (cosmo.H0 * (1 + z_val))
            else:
                D_A.flat[i] = 0
        
        return D_A.reshape(z.shape)
    
    # Тест на поправената функция
    print("\n🧪 ТЕСТ НА ПОПРАВЕНАТА ФУНКЦИЯ:")
    print("-" * 40)
    
    cosmo = NoLambdaCosmology(H0=73.23, Omega_m=0.3046)
    z_test = [1076.8, 1090.0, 100.0, 1.0]
    
    print(f"{'|z':<10} {'Оригинална':<15} {'Поправена':<15} {'Разлика':<15}")
    print("-" * 55)
    
    for z in z_test:
        D_A_original = cosmo.angular_diameter_distance(z)
        D_A_corrected = corrected_angular_diameter_distance(cosmo, z)
        difference = abs(D_A_original - D_A_corrected)
        
        print(f"{z:<10.1f} {D_A_original:<15.2f} {D_A_corrected:<15.2f} {difference:<15.2f}")
    
    return corrected_angular_diameter_distance

def test_theta_s_with_correction():
    """Тест на θ_s с поправената функция"""
    
    print("\n🎯 ТЕСТ НА θ_s С ПОПРАВКАТА")
    print("=" * 60)
    
    cosmo = NoLambdaCosmology(H0=73.23, Omega_m=0.3046)
    
    # Звуков хоризонт
    r_s = cosmo.sound_horizon_scale(z_end=cosmo.z_drag)
    print(f"r_s = {r_s:.2f} Mpc")
    
    # Оригинална D_A
    D_A_original = cosmo.angular_diameter_distance(cosmo.z_drag)
    theta_s_original = r_s / D_A_original
    
    print(f"\nОригинална система:")
    print(f"  D_A(z_drag) = {D_A_original:.2f} Mpc")
    print(f"  θ_s = {theta_s_original:.6f} rad")
    
    # Поправена D_A
    corrected_func = create_corrected_angular_diameter_distance()
    D_A_corrected = corrected_func(cosmo, cosmo.z_drag)
    theta_s_corrected = r_s / D_A_corrected
    
    print(f"\nПоправена система:")
    print(f"  D_A(z_drag) = {D_A_corrected:.2f} Mpc")
    print(f"  θ_s = {theta_s_corrected:.6f} rad")
    
    # Експериментална стойност
    theta_s_obs = 0.010409
    error_original = abs(theta_s_original - theta_s_obs) / theta_s_obs * 100
    error_corrected = abs(theta_s_corrected - theta_s_obs) / theta_s_obs * 100
    
    print(f"\nСравнение с експерименталната стойност:")
    print(f"  θ_s наблюдавано = {theta_s_obs:.6f} rad")
    print(f"  Грешка (оригинал) = {error_original:.1f}%")
    print(f"  Грешка (поправено) = {error_corrected:.1f}%")
    print(f"  Подобрение = {error_original - error_corrected:.1f}%")
    
    return {
        'r_s': r_s,
        'D_A_original': D_A_original,
        'D_A_corrected': D_A_corrected,
        'theta_s_original': theta_s_original,
        'theta_s_corrected': theta_s_corrected,
        'error_original': error_original,
        'error_corrected': error_corrected
    }

if __name__ == "__main__":
    # Основен анализ
    cosmo, z_drag = debug_angular_diameter_calculation()
    
    # Тест на поправката  
    results = test_theta_s_with_correction()
    
    print("\n" + "=" * 60)
    print("✅ АНАЛИЗ ЗАВЪРШЕН!")
    print(f"🎯 Ключов резултат: D_A трябва да е ~{14000:.0f} Mpc, не {results['D_A_original']:.0f} Mpc")
    print("🔧 Необходима корекция на angular_diameter_distance функцията")
    print("=" * 60) 