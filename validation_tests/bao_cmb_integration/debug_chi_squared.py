#!/usr/bin/env python3
"""
Debug Chi-Squared Analysis
==========================

ПРИОРИТЕТ №1: Дебъгване на χ² = 2099.4

Цел: Намиране на грешката в кода, единиците или формулите
която причинява огромния χ².

Стъпки:
1. Изчисляване на χ² стъпка по стъпка за best-fit параметрите
2. Разпечатване на всички междинни стойности
3. Сравняване с известни резултати
4. Идентифициране на проблема
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from typing import Dict, Tuple
import matplotlib.pyplot as plt

# Наши модули
from observational_data import BAOObservationalData, CMBObservationalData
from no_lambda_cosmology import NoLambdaCosmology
from fast_cosmo import *

# Константи
C_KM_S = 299792.458  # km/s
H0_PLANCK = 67.4  # km/s/Mpc
OMEGA_M_PLANCK = 0.315


class ChiSquaredDebugger:
    """
    Детайлен дебъгер за χ² анализ
    """
    
    def __init__(self):
        """Инициализация на дебъгера"""
        
        # Зареждане на данни
        self.bao_data = BAOObservationalData()
        self.cmb_data = CMBObservationalData()
        
        # Best-fit параметри от comprehensive анализа
        self.best_fit_params = {
            'H0': 69.1237,
            'Omega_m': 0.3233,
            'epsilon_bao': 0.0497,
            'epsilon_cmb': 0.0256
        }
        
        # За сравнение - стандартни Planck параметри
        self.planck_params = {
            'H0': 67.4,
            'Omega_m': 0.315,
            'epsilon_bao': 0.0,
            'epsilon_cmb': 0.0
        }
        
        print("🔍 Chi-Squared Debugger инициализиран")
        print(f"🎯 Best-fit params: {self.best_fit_params}")
        
    def debug_full_chi_squared(self):
        """
        Пълен дебъгване на χ² изчислението
        """
        
        print("\n" + "="*60)
        print("🔍 ДЕТАЙЛЕН ДЕБЪГВАНЕ НА χ² ИЗЧИСЛЕНИЕ")
        print("="*60)
        
        # Стъпка 1: Анализ на BAO част
        print("\n📊 СТЪПКА 1: BAO Chi-Squared Analysis")
        bao_chi2, bao_debug = self._debug_bao_chi_squared()
        
        # Стъпка 2: Анализ на CMB част
        print("\n📊 СТЪПКА 2: CMB Chi-Squared Analysis")
        cmb_chi2, cmb_debug = self._debug_cmb_chi_squared()
        
        # Стъпка 3: Общ анализ
        print("\n📊 СТЪПКА 3: Общо Chi-Squared")
        total_chi2 = bao_chi2 + cmb_chi2
        
        print(f"BAO χ²: {bao_chi2:.3f}")
        print(f"CMB χ²: {cmb_chi2:.3f}")
        print(f"ОБЩО χ²: {total_chi2:.3f}")
        
        # Стъпка 4: Сравнение с очакванията
        print("\n📊 СТЪПКА 4: Сравнение с очаквания")
        self._compare_with_expectations(bao_chi2, cmb_chi2, total_chi2)
        
        # Стъпка 5: Детайлен анализ
        print("\n📊 СТЪПКА 5: Детайлен анализ")
        self._detailed_analysis(bao_debug, cmb_debug)
        
        return total_chi2, bao_debug, cmb_debug
    
    def _debug_bao_chi_squared(self) -> Tuple[float, Dict]:
        """
        Детайлен дебъгване на BAO χ²
        """
        
        # Получаване на BAO данни
        bao_combined = self.bao_data.get_combined_data()
        z_bao = bao_combined['redshifts']
        DV_rs_obs = bao_combined['DV_rs']
        DV_rs_err = bao_combined['DV_rs_err']
        
        print(f"📊 BAO данни: {len(z_bao)} точки")
        print(f"📊 Redshifts: {z_bao}")
        print(f"📊 DV/rs_obs: {DV_rs_obs}")
        print(f"📊 DV/rs_err: {DV_rs_err}")
        
        # Създаване на космологичен модел
        cosmo = NoLambdaCosmology(**self.best_fit_params)
        
        # Изчисляване на sound horizon
        print(f"\n🔍 Изчисляване на sound horizon...")
        r_s = cosmo.sound_horizon_scale()
        print(f"r_s = {r_s:.3f} Mpc")
        
        # Проверка дали r_s е разумен
        if r_s < 100 or r_s > 200:
            print(f"⚠️ WARNING: r_s = {r_s:.3f} Mpc изглежда необичайно!")
            print(f"⚠️ Очакваната стойност е ~147 Mpc")
        
        # Изчисляване на теоретичните DV/rs стойности
        print(f"\n🔍 Изчисляване на теоретични DV/rs...")
        
        DV_rs_theory = []
        debug_info = []
        
        for i, z in enumerate(z_bao):
            print(f"\n--- z = {z:.3f} ---")
            
            # Angular diameter distance
            D_A = cosmo.angular_diameter_distance(z)
            print(f"D_A({z:.3f}) = {D_A:.3f} Mpc")
            
            # Hubble parameter
            H_z = cosmo.hubble_function(z)
            print(f"H({z:.3f}) = {H_z:.3f} km/s/Mpc")
            
            # Hubble distance
            D_H = C_KM_S / H_z
            print(f"D_H({z:.3f}) = {D_H:.3f} Mpc")
            
            # Dilation scale D_V
            D_V = (z * D_A**2 * D_H)**(1/3)
            print(f"D_V({z:.3f}) = {D_V:.3f} Mpc")
            
            # DV/rs
            DV_rs = D_V / r_s
            print(f"DV/rs({z:.3f}) = {DV_rs:.6f}")
            
            DV_rs_theory.append(DV_rs)
            
            # Дебъгване информация
            debug_info.append({
                'z': z,
                'D_A': D_A,
                'H_z': H_z,
                'D_H': D_H,
                'D_V': D_V,
                'DV_rs_theory': DV_rs,
                'DV_rs_obs': DV_rs_obs[i],
                'DV_rs_err': DV_rs_err[i]
            })
        
        DV_rs_theory = np.array(DV_rs_theory)
        
        # Изчисляване на остатъците
        print(f"\n🔍 Изчисляване на остатъци...")
        residuals = DV_rs_obs - DV_rs_theory
        
        print(f"Residuals: {residuals}")
        relative_residuals = residuals/DV_rs_obs*100
        print(f"Relative residuals (%): {relative_residuals}")
        
        # Индивидуални приноси към χ²
        print(f"\n🔍 Индивидуални приноси към χ²...")
        individual_chi2 = (residuals / DV_rs_err)**2
        
        for i, (z, chi2_i) in enumerate(zip(z_bao, individual_chi2)):
            print(f"χ²({z:.3f}) = {chi2_i:.3f}")
        
        # Общо BAO χ²
        bao_chi2 = np.sum(individual_chi2)
        print(f"\n📊 BAO χ² = {bao_chi2:.3f}")
        
        # 🚨 КРИТИЧНО: Анализ на огромната разлика
        print(f"\n🚨 КРИТИЧНО: Анализ на разликата")
        print(f"Obs range: {np.min(DV_rs_obs):.1f} - {np.max(DV_rs_obs):.1f}")
        print(f"Theory range: {np.min(DV_rs_theory):.3f} - {np.max(DV_rs_theory):.3f}")
        print(f"Ratio factor: {np.mean(DV_rs_obs/DV_rs_theory):.1f}")
        
        return bao_chi2, {
            'r_s': r_s,
            'debug_info': debug_info,
            'residuals': residuals,
            'individual_chi2': individual_chi2,
            'total_chi2': bao_chi2
        }
    
    def _debug_cmb_chi_squared(self) -> Tuple[float, Dict]:
        """
        Детайлен дебъгване на CMB χ²
        """
        
        # Получаване на CMB данни
        acoustic_data = self.cmb_data.get_acoustic_scale()
        theta_s_obs = acoustic_data['theta_s']
        theta_s_err = acoustic_data['theta_s_err']
        
        print(f"📊 CMB theta_s_obs: {theta_s_obs:.6f}")
        print(f"📊 CMB theta_s_err: {theta_s_err:.6f}")
        
        # Създаване на космологичен модел
        cosmo = NoLambdaCosmology(**self.best_fit_params)
        
        # Изчисляване на теоретичен theta_s
        print(f"\n🔍 Изчисляване на теоретичен theta_s...")
        
        # Sound horizon
        r_s = cosmo.sound_horizon_scale()
        print(f"r_s = {r_s:.3f} Mpc")
        
        # Comoving distance към decoupling (z ~ 1090)
        z_cmb = 1090
        D_M = cosmo.comoving_distance(z_cmb)
        print(f"D_M({z_cmb}) = {D_M:.3f} Mpc")
        
        # Angular scale
        theta_s_theory = r_s / D_M
        print(f"theta_s_theory = {theta_s_theory:.6f}")
        
        # Проверка дали theta_s е разумен
        if theta_s_theory < 0.0008 or theta_s_theory > 0.0015:
            print(f"⚠️ WARNING: theta_s = {theta_s_theory:.6f} изглежда необичайно!")
            print(f"⚠️ Очакваната стойност е ~0.0104")
        
        # Остатък
        residual = theta_s_obs - theta_s_theory
        print(f"Residual: {residual:.6f}")
        relative_residual = residual/theta_s_obs*100
        print(f"Relative residual (%): {relative_residual:.2f}")
        
        # CMB χ²
        cmb_chi2 = (residual / theta_s_err)**2
        print(f"\n📊 CMB χ² = {cmb_chi2:.3f}")
        
        # 🚨 КРИТИЧНО: Анализ на CMB разликата
        print(f"\n🚨 КРИТИЧНО: CMB анализ")
        print(f"Obs theta_s: {theta_s_obs:.6f}")
        print(f"Theory theta_s: {theta_s_theory:.6f}")
        print(f"Ratio: {theta_s_obs/theta_s_theory:.2f}")
        
        return cmb_chi2, {
            'theta_s_obs': theta_s_obs,
            'theta_s_theory': theta_s_theory,
            'theta_s_err': theta_s_err,
            'residual': residual,
            'chi2': cmb_chi2
        }
    
    def _compare_with_expectations(self, bao_chi2: float, cmb_chi2: float, total_chi2: float):
        """
        Сравнение с очакванията
        """
        
        # Броя на данните
        bao_combined = self.bao_data.get_combined_data()
        n_bao = len(bao_combined['redshifts'])
        n_cmb = 4  # Приблизително (theta_s + peaks)
        n_total = n_bao + n_cmb
        
        n_params = len(self.best_fit_params)
        dof = n_total - n_params
        
        print(f"📊 Брой BAO точки: {n_bao}")
        print(f"📊 Брой CMB constrains: {n_cmb}")
        print(f"📊 Общо данни: {n_total}")
        print(f"📊 Параметри: {n_params}")
        print(f"📊 DOF: {dof}")
        
        # Очаквания
        expected_chi2 = dof
        reduced_chi2 = total_chi2 / dof
        
        print(f"\n📊 Очакван χ² ≈ {expected_chi2}")
        print(f"📊 Намерен χ² = {total_chi2:.1f}")
        print(f"📊 Reduced χ² = {reduced_chi2:.1f}")
        
        # Диагноза
        if reduced_chi2 > 10:
            print(f"🚨 КРИТИЧНО: Reduced χ² >> 1 - сигурно има грешка в кода!")
        elif reduced_chi2 > 3:
            print(f"⚠️ ПРОБЛЕМ: Reduced χ² > 3 - вероятно има грешка")
        elif reduced_chi2 > 1.5:
            print(f"⚠️ ВНИМАНИЕ: Reduced χ² > 1.5 - възможен проблем")
        else:
            print(f"✅ ДОБРЕ: Reduced χ² ≈ 1")
    
    def _detailed_analysis(self, bao_debug: Dict, cmb_debug: Dict):
        """
        Детайлен анализ за намиране на проблема
        """
        
        print("\n🔍 ДЕТАЙЛЕН АНАЛИЗ:")
        
        # Анализ на BAO
        print("\n--- BAO АНАЛИЗ ---")
        r_s = bao_debug['r_s']
        
        # Проверка на sound horizon
        print(f"Sound horizon: {r_s:.3f} Mpc")
        if r_s < 100:
            print("❌ ПРОБЛЕМ: r_s е твърде малък!")
        elif r_s > 200:
            print("❌ ПРОБЛЕМ: r_s е твърде голям!")
        else:
            print("✅ r_s изглежда разумен")
        
        # Анализ на най-големите приноси
        individual_chi2 = bao_debug['individual_chi2']
        max_chi2_idx = np.argmax(individual_chi2)
        max_chi2 = individual_chi2[max_chi2_idx]
        
        print(f"Най-голям принос към χ²: {max_chi2:.3f} (index {max_chi2_idx})")
        
        if max_chi2 > 100:
            print("❌ ПРОБЛЕМ: Една точка дава огромен принос!")
            debug_point = bao_debug['debug_info'][max_chi2_idx]
            print(f"Проблемна точка: z={debug_point['z']:.3f}")
            print(f"Theory: {debug_point['DV_rs_theory']:.6f}")
            print(f"Obs: {debug_point['DV_rs_obs']:.6f}")
            print(f"Error: {debug_point['DV_rs_err']:.6f}")
        
        # Анализ на CMB
        print("\n--- CMB АНАЛИЗ ---")
        theta_s_theory = cmb_debug['theta_s_theory']
        theta_s_obs = cmb_debug['theta_s_obs']
        
        print(f"CMB theta_s theory: {theta_s_theory:.6f}")
        print(f"CMB theta_s obs: {theta_s_obs:.6f}")
        
        relative_error = abs(theta_s_theory - theta_s_obs) / theta_s_obs * 100
        print(f"Relative error: {relative_error:.1f}%")
        
        if relative_error > 50:
            print("❌ ПРОБЛЕМ: Огромна разлика в CMB!")
        elif relative_error > 10:
            print("⚠️ ПРОБЛЕМ: Голяма разлика в CMB")
        else:
            print("✅ CMB изглежда разумно")
    
    def compare_with_reference(self):
        """
        Сравнение с референтни стойности (Planck, литература)
        """
        
        print("\n" + "="*60)
        print("🔍 СРАВНЕНИЕ С РЕФЕРЕНТНИ СТОЙНОСТИ")
        print("="*60)
        
        # Референтни стойности от Planck 2018
        print("\n📚 Референтни стойности (Planck 2018):")
        print(f"H0 = 67.4 km/s/Mpc")
        print(f"Omega_m = 0.315")
        print(f"r_s ≈ 147 Mpc")
        print(f"theta_s ≈ 0.0104")
        
        # Нашите стойности
        cosmo = NoLambdaCosmology(**self.best_fit_params)
        r_s = cosmo.sound_horizon_scale()
        theta_s = cosmo.cmb_angular_scale()
        
        print(f"\n📊 Нашите стойности:")
        print(f"H0 = {self.best_fit_params['H0']:.1f} km/s/Mpc")
        print(f"Omega_m = {self.best_fit_params['Omega_m']:.3f}")
        print(f"r_s = {r_s:.1f} Mpc")
        print(f"theta_s = {theta_s:.6f}")
        
        # Сравнение
        print(f"\n📊 Сравнение:")
        h0_diff = abs(self.best_fit_params['H0'] - 67.4) / 67.4 * 100
        omega_diff = abs(self.best_fit_params['Omega_m'] - 0.315) / 0.315 * 100
        rs_diff = abs(r_s - 147) / 147 * 100
        theta_diff = abs(theta_s - 0.0104) / 0.0104 * 100
        
        print(f"H0 difference: {h0_diff:.1f}%")
        print(f"Omega_m difference: {omega_diff:.1f}%")
        print(f"r_s difference: {rs_diff:.1f}%")
        print(f"theta_s difference: {theta_diff:.1f}%")
        
        if any([h0_diff > 20, omega_diff > 20, rs_diff > 20, theta_diff > 20]):
            print("❌ ПРОБЛЕМ: Стойностите са твърде различни от очакваните!")
        else:
            print("✅ Стойностите изглеждат разумни")
    
    def save_debug_results(self, debug_results: Dict):
        """
        Записване на дебъгване резултатите
        """
        
        print("\n💾 Записване на дебъгване резултати...")
        
        # Създаване на DataFrame за BAO
        bao_data = []
        for info in debug_results[1]['debug_info']:
            bao_data.append({
                'z': info['z'],
                'DV_rs_obs': info['DV_rs_obs'],
                'DV_rs_theory': info['DV_rs_theory'],
                'DV_rs_err': info['DV_rs_err'],
                'residual': info['DV_rs_obs'] - info['DV_rs_theory'],
                'chi2_contribution': ((info['DV_rs_obs'] - info['DV_rs_theory']) / info['DV_rs_err'])**2
            })
        
        bao_df = pd.DataFrame(bao_data)
        bao_df.to_csv('debug_bao_analysis.csv', index=False)
        
        # Записване на summary
        with open('debug_chi_squared_summary.txt', 'w', encoding='utf-8') as f:
            f.write("CHI-SQUARED DEBUG SUMMARY\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("BEST-FIT PARAMETERS:\n")
            for param, value in self.best_fit_params.items():
                f.write(f"{param}: {value:.6f}\n")
            
            f.write(f"\nCHI-SQUARED BREAKDOWN:\n")
            f.write(f"BAO χ²: {debug_results[1]['total_chi2']:.3f}\n")
            f.write(f"CMB χ²: {debug_results[2]['chi2']:.3f}\n")
            f.write(f"Total χ²: {debug_results[0]:.3f}\n")
            
            f.write(f"\nKEY QUANTITIES:\n")
            f.write(f"Sound horizon: {debug_results[1]['r_s']:.3f} Mpc\n")
            f.write(f"CMB theta_s: {debug_results[2]['theta_s_theory']:.6f}\n")
        
        print("✅ Дебъгване файлове записани:")
        print("   📊 debug_bao_analysis.csv")
        print("   📋 debug_chi_squared_summary.txt")


def test_standard_lcdm_comparison():
    """Тест със стандартен ΛCDM модел за сравнение"""
    print("\n🔍 ТЕСТ СЪС СТАНДАРТЕН ΛCDM МОДЕЛ")
    print("=" * 40)
    
    # Стандартен ΛCDM модел (Planck 2018)
    from astropy.cosmology import Planck18
    cosmo = Planck18
    
    # Нашите параметри
    H0 = 69.1  # km/s/Mpc
    Omega_m = 0.3233
    Omega_b = 0.049
    
    print(f"Планк 2018 r_s: {cosmo.comoving_distance(1090).value:.1f} Mpc (за z=1090)")
    print(f"Планк 2018 H0: {cosmo.H0.value:.1f} km/s/Mpc")
    print(f"Планк 2018 Omega_m: {cosmo.Om0:.3f}")
    print(f"Планк 2018 theta_s: {cosmo.angular_diameter_distance(1090).value / cosmo.comoving_distance(1090).value:.6f}")
    
    # Опростен тест без анизотропни корекции
    print("\n🔍 ОПРОСТЕН ТЕСТ БЕЗ АНИЗОТРОПНИ КОРЕКЦИИ")
    print("=" * 40)
    
    # Стандартен E(z) за No-Lambda
    def E_function_simple(z, H0, Omega_m, Omega_r=8.24e-5):
        """Опростена E(z) функция БЕЗ анизотропни корекции"""
        Omega_k = 1 - Omega_m - Omega_r
        return np.sqrt(Omega_m*(1+z)**3 + Omega_r*(1+z)**4 + Omega_k*(1+z)**2)
    
    def sound_speed_simple(z, Omega_b=0.049):
        """Опростена sound speed БЕЗ анизотропни корекции"""
        c = 299792458  # m/s
        T_cmb = 2.725  # K
        T_nu = T_cmb * (4/11)**(1/3)  # K
        Omega_gamma = 8.24e-5 * (8/7) * (T_cmb/T_nu)**4
        R_b = (3 * Omega_b) / (4 * Omega_gamma * (1 + z))
        return c / np.sqrt(3 * (1 + R_b))
    
    def hubble_simple(z, H0, Omega_m, Omega_r=8.24e-5):
        """Опростена Hubble функция БЕЗ анизотропни корекции"""
        return H0 * E_function_simple(z, H0, Omega_m, Omega_r)
    
    def sound_horizon_simple(z_drag, H0, Omega_m, Omega_b):
        """Опростен sound horizon БЕЗ анизотропни корекции"""
        def integrand(z):
            c_s = sound_speed_simple(z, Omega_b)  # m/s
            H_z = hubble_simple(z, H0, Omega_m)  # km/s/Mpc
            return (c_s / 1000) / H_z  # Mpc
        
        from scipy.integrate import quad
        r_s, _ = quad(integrand, 0, z_drag)
        return r_s
    
    # Drag epoch с стандартна формула
    def drag_epoch_simple(H0, Omega_m, Omega_b):
        """Опростена drag epoch формула"""
        b1 = 0.313 * (Omega_m * H0**2 / 100)**(-0.419) * (1 + 0.607 * (Omega_m * H0**2 / 100)**0.674)
        b2 = 0.238 * (Omega_m * H0**2 / 100)**0.223
        z_drag = 1291 * (Omega_m * H0**2 / 100)**0.251 / (1 + 0.659 * (Omega_m * H0**2 / 100)**0.828) * (1 + b1 * (Omega_b * H0**2 / 100)**b2)
        return z_drag
    
    # Изчисления
    z_drag_simple = drag_epoch_simple(H0, Omega_m, Omega_b)
    r_s_simple = sound_horizon_simple(z_drag_simple, H0, Omega_m, Omega_b)
    
    print(f"Опростен z_drag: {z_drag_simple:.1f}")
    print(f"Опростен r_s: {r_s_simple:.1f} Mpc")
    
    # Сравнение с нашия модел
    print(f"\nСравнение:")
    print(f"Наш модел z_drag: {598.5:.1f}")
    print(f"Наш модел r_s: {1966.151:.1f} Mpc")
    print(f"Опростен z_drag: {z_drag_simple:.1f}")
    print(f"Опростен r_s: {r_s_simple:.1f} Mpc")
    
    print(f"\nПроблем: Нашият r_s е {1966.151/r_s_simple:.1f}x по-голям от опростения!")
    
    return z_drag_simple, r_s_simple


def main():
    """
    Главна функция за дебъгване
    """
    
    print("🚨 CHI-SQUARED DEBUGGER")
    print("🎯 Цел: Намиране на грешката в χ² = 2099.4")
    print("=" * 50)
    
    # Тест със стандартен ΛCDM за сравнение
    test_standard_lcdm_comparison()
    
    # Обичайното дебъгване
    analyzer = ChiSquaredDebugger()
    analyzer.debug_full_chi_squared()
    
    print("\n🎉 Дебъгването завърши!")
    print("📋 Проверете записаните файлове за детайли")


if __name__ == "__main__":
    main() 