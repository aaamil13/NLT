#!/usr/bin/env python3
"""
Анализ на BAO и CMB параметри без тъмна енергия (Λ=0)

Този скрипт сравнява:
1. Стандартен ΛCDM модел
2. Модел без тъмна енергия (Λ=0)
3. Анизотропен модел без тъмна енергия
4. Реални наблюдателни данни

Фокусира се на:
- BAO скала на звуковия хоризонт r_s
- CMB ъглова скала θ_s и позиция на първия пик l_peak
- Анизотропни корекции и посочни вариации
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import warnings

# Импортиране на модулите
from no_lambda_cosmology import NoLambdaCosmology
from anisotropic_nonlinear_time import AnisotropicNonlinearTimeCosmology

# Настройка на стиловете
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Настройка на логирането
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Физични константи
c = 299792458  # м/с


class StandardLCDM:
    """Стандартен ΛCDM модел за сравнение"""
    
    def __init__(self, H0=67.4, Omega_m=0.315, Omega_Lambda=0.685, Omega_b=0.049):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        self.Omega_b = Omega_b
        self.Omega_r = 8.24e-5
        
        # Фиксирани стойности за z_drag и z_star
        self.z_drag = 1060
        self.z_star = 1090
        
    def hubble_function(self, z):
        """Стандартна ΛCDM Хъбъл функция"""
        z = np.asarray(z)
        return self.H0 * np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_Lambda)
    
    def sound_horizon_scale(self):
        """Стандартна BAO скала (Planck 2018)"""
        return 147.09  # Mpc
    
    def angular_diameter_distance(self, z):
        """Стандартно ъглово разстояние"""
        z = np.asarray(z)
        
        def integrand(z_val):
            return c / (self.hubble_function(z_val) * 1000)
        
        D_A = np.zeros_like(z)
        for i, z_val in enumerate(z.flat):
            if z_val > 0:
                integral, _ = integrate.quad(integrand, 0, z_val)
                D_A.flat[i] = integral / (1 + z_val)
            else:
                D_A.flat[i] = 0
        
        return D_A.reshape(z.shape)
    
    def cmb_angular_scale(self):
        """Стандартна CMB ъглова скала"""
        r_s = self.sound_horizon_scale()
        D_A_star = self.angular_diameter_distance(self.z_star)
        return r_s / D_A_star
    
    def cmb_peak_position(self):
        """Стандартна позиция на първия CMB пик"""
        theta_s = self.cmb_angular_scale()
        return np.pi / theta_s


def compare_cosmological_models():
    """Сравнение на космологичните модели"""
    
    print("🌌 СРАВНЕНИЕ НА КОСМОЛОГИЧНИТЕ МОДЕЛИ")
    print("=" * 80)
    
    # Създаване на моделите
    lcdm = StandardLCDM()
    no_lambda = NoLambdaCosmology(epsilon_bao=0.02, epsilon_cmb=0.015)
    
    # Диагностики на моделите
    diag_no_lambda = no_lambda.diagnostics()
    
    # Създаване на сравнителна таблица
    print(f"\n📊 ОСНОВНИ ПАРАМЕТРИ:")
    print(f"{'Параметър':<25} {'ΛCDM':<15} {'No-Λ':<15} {'Разлика':<15}")
    print("-" * 75)
    
    # Основни космологични параметри
    print(f"{'Ωₘ':<25} {lcdm.Omega_m:<15.4f} {no_lambda.Omega_m:<15.4f} {no_lambda.Omega_m - lcdm.Omega_m:<15.4f}")
    print(f"{'ΩΛ':<25} {lcdm.Omega_Lambda:<15.4f} {no_lambda.Omega_Lambda:<15.4f} {no_lambda.Omega_Lambda - lcdm.Omega_Lambda:<15.4f}")
    print(f"{'Ωₖ':<25} {0.0:<15.4f} {diag_no_lambda['Omega_k']:<15.4f} {diag_no_lambda['Omega_k']:<15.4f}")
    
    print(f"\n🔍 КРИТИЧНИ ЧЕРВЕНИ ОТМЕСТВАНИЯ:")
    print(f"{'z_drag':<25} {lcdm.z_drag:<15.1f} {diag_no_lambda['z_drag']:<15.1f} {diag_no_lambda['z_drag'] - lcdm.z_drag:<15.1f}")
    print(f"{'z_star':<25} {lcdm.z_star:<15.1f} {diag_no_lambda['z_star']:<15.1f} {diag_no_lambda['z_star'] - lcdm.z_star:<15.1f}")
    
    print(f"\n🎵 BAO ПАРАМЕТРИ:")
    r_s_lcdm = lcdm.sound_horizon_scale()
    r_s_no_lambda = diag_no_lambda['r_s_isotropic']
    print(f"{'r_s [Mpc]':<25} {r_s_lcdm:<15.3f} {r_s_no_lambda:<15.3f} {r_s_no_lambda - r_s_lcdm:<15.3f}")
    print(f"{'Относителна разлика':<25} {'-':<15} {(r_s_no_lambda - r_s_lcdm)/r_s_lcdm*100:<15.2f}% {'-':<15}")
    
    print(f"\n🌌 CMB ПАРАМЕТРИ:")
    theta_s_lcdm = lcdm.cmb_angular_scale()
    theta_s_no_lambda = diag_no_lambda['theta_s_isotropic']
    l_peak_lcdm = lcdm.cmb_peak_position()
    l_peak_no_lambda = diag_no_lambda['l_peak_isotropic']
    
    print(f"{'θ_s [rad]':<25} {theta_s_lcdm:<15.6f} {theta_s_no_lambda:<15.6f} {theta_s_no_lambda - theta_s_lcdm:<15.6f}")
    print(f"{'l_peak':<25} {l_peak_lcdm:<15.1f} {l_peak_no_lambda:<15.1f} {l_peak_no_lambda - l_peak_lcdm:<15.1f}")
    
    print(f"\n⏰ ВЪЗРАСТ НА ВСЕЛЕНАТА:")
    age_lcdm = 13.8  # Gyr (стандартна стойност)
    age_no_lambda = diag_no_lambda['age_universe_Gyr']
    print(f"{'Възраст [Gyr]':<25} {age_lcdm:<15.2f} {age_no_lambda:<15.2f} {age_no_lambda - age_lcdm:<15.2f}")
    
    return lcdm, no_lambda, diag_no_lambda


def analyze_bao_effects():
    """Анализ на BAO ефектите без тъмна енергия"""
    
    print(f"\n🎵 ДЕТАЙЛЕН BAO АНАЛИЗ")
    print("=" * 50)
    
    # Създаване на моделите
    lcdm = StandardLCDM()
    no_lambda = NoLambdaCosmology(epsilon_bao=0.02)
    
    # Създаване на графики
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BAO Анализ: Ефекти на липсата на тъмна енергия', fontsize=16)
    
    # Subplot 1: Хъбъл функции
    ax1 = axes[0, 0]
    ax1.set_title('Хъбъл функции H(z)')
    
    z_range = np.logspace(-2, 0.5, 100)
    H_lcdm = lcdm.hubble_function(z_range)
    H_no_lambda = no_lambda.hubble_function(z_range)
    
    ax1.plot(z_range, H_lcdm, '--', label='ΛCDM', linewidth=2, color='black')
    ax1.plot(z_range, H_no_lambda, '-', label='No-Λ', linewidth=2, color='blue')
    
    ax1.set_xlabel('Червено отместване z')
    ax1.set_ylabel('H(z) [km/s/Mpc]')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Относителни разлики
    ax2 = axes[0, 1]
    ax2.set_title('Относителни разлики в H(z)')
    
    relative_diff = (H_no_lambda - H_lcdm) / H_lcdm * 100
    ax2.plot(z_range, relative_diff, '-', linewidth=2, color='red')
    
    ax2.set_xlabel('Червено отместване z')
    ax2.set_ylabel('ΔH/H_ΛCDM [%]')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Subplot 3: Скорост на звука
    ax3 = axes[1, 0]
    ax3.set_title('Скорост на звука c_s(z)')
    
    z_early = np.logspace(1, 3, 100)  # z от 10 до 1000
    c_s_values = no_lambda.sound_speed(z_early) / c  # Нормализирано към c
    
    ax3.plot(z_early, c_s_values, '-', linewidth=2, color='green')
    ax3.set_xlabel('Червено отместване z')
    ax3.set_ylabel('c_s/c')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Интегранд на звуковия хоризонт
    ax4 = axes[1, 1]
    ax4.set_title('Интегранд на звуковия хоризонт')
    
    integrand_values = []
    for z_val in z_early:
        integrand_val = no_lambda.sound_horizon_integrand(z_val)
        integrand_values.append(integrand_val)
    
    ax4.plot(z_early, integrand_values, '-', linewidth=2, color='purple')
    ax4.set_xlabel('Червено отместване z')
    ax4.set_ylabel('c_s(z)/H(z) [Mpc]')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bao_analysis_no_lambda.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Количествени резултати
    print(f"\n📈 BAO КОЛИЧЕСТВЕНИ РЕЗУЛТАТИ:")
    print(f"  ΛCDM r_s = {lcdm.sound_horizon_scale():.3f} Mpc")
    print(f"  No-Λ r_s = {no_lambda.sound_horizon_scale():.3f} Mpc")
    print(f"  Разлика = {no_lambda.sound_horizon_scale() - lcdm.sound_horizon_scale():.3f} Mpc")
    print(f"  Относителна разлика = {(no_lambda.sound_horizon_scale() - lcdm.sound_horizon_scale())/lcdm.sound_horizon_scale()*100:.2f}%")


def analyze_cmb_effects():
    """Анализ на CMB ефектите без тъмна енергия"""
    
    print(f"\n🌌 ДЕТАЙЛЕН CMB АНАЛИЗ")
    print("=" * 50)
    
    # Създаване на моделите
    lcdm = StandardLCDM()
    no_lambda = NoLambdaCosmology(epsilon_cmb=0.015)
    
    # Създаване на графики
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CMB Анализ: Ефекти на липсата на тъмна енергия', fontsize=16)
    
    # Subplot 1: Ъглови разстояния
    ax1 = axes[0, 0]
    ax1.set_title('Ъглови диаметрови разстояния')
    
    z_range = np.logspace(-1, 3.5, 100)  # z от 0.1 до ~3162
    D_A_lcdm = lcdm.angular_diameter_distance(z_range)
    D_A_no_lambda = no_lambda.angular_diameter_distance(z_range)
    
    ax1.plot(z_range, D_A_lcdm, '--', label='ΛCDM', linewidth=2, color='black')
    ax1.plot(z_range, D_A_no_lambda, '-', label='No-Λ', linewidth=2, color='blue')
    
    # Маркиране на z_star
    ax1.axvline(x=lcdm.z_star, color='red', linestyle=':', alpha=0.7, label='z* (ΛCDM)')
    ax1.axvline(x=no_lambda.z_star, color='orange', linestyle=':', alpha=0.7, label='z* (No-Λ)')
    
    ax1.set_xlabel('Червено отместване z')
    ax1.set_ylabel('D_A(z) [Mpc]')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Относителни разлики в D_A
    ax2 = axes[0, 1]
    ax2.set_title('Относителни разлики в D_A(z)')
    
    relative_diff_DA = (D_A_no_lambda - D_A_lcdm) / D_A_lcdm * 100
    ax2.plot(z_range, relative_diff_DA, '-', linewidth=2, color='red')
    
    ax2.set_xlabel('Червено отместване z')
    ax2.set_ylabel('ΔD_A/D_A_ΛCDM [%]')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Subplot 3: CMB мощностен спектър (симулация)
    ax3 = axes[1, 0]
    ax3.set_title('CMB мощностен спектър (симулация)')
    
    # Симулация на CMB пикове
    l_values = np.logspace(0.5, 3, 1000)  # l от ~3 до 1000
    
    # Приблизителен CMB спектър (Gaussian пикове)
    def cmb_spectrum_approx(l, l_peak, amplitude=1.0):
        """Приблизителен CMB спектър с Gaussian пик"""
        return amplitude * np.exp(-0.5 * ((l - l_peak) / (l_peak * 0.2))**2)
    
    # Пикове за различните модели
    l_peak_lcdm = lcdm.cmb_peak_position()
    l_peak_no_lambda = no_lambda.cmb_peak_position()
    
    C_l_lcdm = cmb_spectrum_approx(l_values, l_peak_lcdm)
    C_l_no_lambda = cmb_spectrum_approx(l_values, l_peak_no_lambda)
    
    ax3.plot(l_values, C_l_lcdm, '--', label=f'ΛCDM (l_peak={l_peak_lcdm:.0f})', linewidth=2, color='black')
    ax3.plot(l_values, C_l_no_lambda, '-', label=f'No-Λ (l_peak={l_peak_no_lambda:.0f})', linewidth=2, color='blue')
    
    ax3.set_xlabel('Мултипол l')
    ax3.set_ylabel('C_l [произволни единици]')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Измествания на CMB пиковете
    ax4 = axes[1, 1]
    ax4.set_title('Измествания на CMB пиковете')
    
    # Първи няколко пика
    peak_numbers = np.array([1, 2, 3, 4, 5])
    l_peaks_lcdm = l_peak_lcdm * peak_numbers
    l_peaks_no_lambda = l_peak_no_lambda * peak_numbers
    
    shift = l_peaks_no_lambda - l_peaks_lcdm
    
    ax4.plot(peak_numbers, shift, 'o-', linewidth=2, markersize=8, color='red')
    ax4.set_xlabel('Номер на пика')
    ax4.set_ylabel('Измествоне на пика Δl')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('cmb_analysis_no_lambda.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Количествени резултати
    print(f"\n📈 CMB КОЛИЧЕСТВЕНИ РЕЗУЛТАТИ:")
    print(f"  ΛCDM θ_s = {lcdm.cmb_angular_scale():.6f} rad")
    print(f"  No-Λ θ_s = {no_lambda.cmb_angular_scale():.6f} rad")
    print(f"  Разлика = {no_lambda.cmb_angular_scale() - lcdm.cmb_angular_scale():.6f} rad")
    print(f"  ΛCDM l_peak = {lcdm.cmb_peak_position():.1f}")
    print(f"  No-Λ l_peak = {no_lambda.cmb_peak_position():.1f}")
    print(f"  Измествоне = {no_lambda.cmb_peak_position() - lcdm.cmb_peak_position():.1f}")


def analyze_anisotropic_effects():
    """Анализ на анизотропните ефекти в No-Λ модела"""
    
    print(f"\n🧭 АНАЛИЗ НА АНИЗОТРОПНИТЕ ЕФЕКТИ")
    print("=" * 50)
    
    # Създаване на модел с различни степени на анизотропия
    anisotropy_levels = [
        (0.0, 0.0, "Изотропно"),
        (0.01, 0.008, "Слаба анизотропия"),
        (0.03, 0.02, "Умерена анизотропия"),
        (0.06, 0.04, "Силна анизотропия")
    ]
    
    # Тестови посоки
    directions = [
        (0, 0, "Полярна"),
        (np.pi/2, 0, "Екваториална-X"),
        (np.pi/2, np.pi/2, "Екваториална-Y"),
        (np.pi/4, np.pi/4, "Диагонална")
    ]
    
    # Създаване на графики
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Анизотропни ефекти в No-Λ модела', fontsize=16)
    
    # Subplot 1: BAO анизотропия
    ax1 = axes[0, 0]
    ax1.set_title('BAO анизотропия (r_s по посоки)')
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (eps_bao, eps_cmb, label) in enumerate(anisotropy_levels[1:]):
        cosmo = NoLambdaCosmology(epsilon_bao=eps_bao, epsilon_cmb=eps_cmb)
        
        r_s_values = []
        dir_names = []
        
        for theta, phi, dir_name in directions:
            r_s = cosmo.sound_horizon_scale(theta=theta, phi=phi)
            r_s_values.append(r_s)
            dir_names.append(dir_name)
        
        ax1.plot(range(len(directions)), r_s_values, 'o-', 
                label=label, color=colors[i], linewidth=2, markersize=8)
    
    ax1.set_xticks(range(len(directions)))
    ax1.set_xticklabels([d[2] for d in directions], rotation=45)
    ax1.set_ylabel('r_s [Mpc]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: CMB анизотропия
    ax2 = axes[0, 1]
    ax2.set_title('CMB анизотропия (l_peak по посоки)')
    
    for i, (eps_bao, eps_cmb, label) in enumerate(anisotropy_levels[1:]):
        cosmo = NoLambdaCosmology(epsilon_bao=eps_bao, epsilon_cmb=eps_cmb)
        
        l_peak_values = []
        
        for theta, phi, dir_name in directions:
            l_peak = cosmo.cmb_peak_position(theta=theta, phi=phi)
            l_peak_values.append(l_peak)
        
        ax2.plot(range(len(directions)), l_peak_values, 'o-', 
                label=label, color=colors[i], linewidth=2, markersize=8)
    
    ax2.set_xticks(range(len(directions)))
    ax2.set_xticklabels([d[2] for d in directions], rotation=45)
    ax2.set_ylabel('l_peak')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Анизотропия като функция от силата
    ax3 = axes[1, 0]
    ax3.set_title('Анизотропия като функция от параметъра')
    
    epsilon_range = np.linspace(0, 0.08, 20)
    r_s_variations = []
    l_peak_variations = []
    
    for eps in epsilon_range:
        cosmo = NoLambdaCosmology(epsilon_bao=eps, epsilon_cmb=eps*0.7)
        
        r_s_values = []
        l_peak_values = []
        
        for theta, phi, _ in directions:
            r_s_values.append(cosmo.sound_horizon_scale(theta=theta, phi=phi))
            l_peak_values.append(cosmo.cmb_peak_position(theta=theta, phi=phi))
        
        r_s_variations.append(np.std(r_s_values) / np.mean(r_s_values) * 100)
        l_peak_variations.append(np.std(l_peak_values) / np.mean(l_peak_values) * 100)
    
    ax3.plot(epsilon_range, r_s_variations, '-', label='BAO (r_s)', linewidth=2, color='blue')
    ax3.plot(epsilon_range, l_peak_variations, '-', label='CMB (l_peak)', linewidth=2, color='red')
    
    ax3.set_xlabel('Параметър на анизотропия ε')
    ax3.set_ylabel('Коефициент на вариация [%]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Корелация BAO-CMB
    ax4 = axes[1, 1]
    ax4.set_title('Корелация между BAO и CMB анизотропии')
    
    # Данни за корелацията
    cosmo = NoLambdaCosmology(epsilon_bao=0.04, epsilon_cmb=0.03)
    
    r_s_dir = []
    l_peak_dir = []
    
    for theta, phi, _ in directions:
        r_s_dir.append(cosmo.sound_horizon_scale(theta=theta, phi=phi))
        l_peak_dir.append(cosmo.cmb_peak_position(theta=theta, phi=phi))
    
    ax4.scatter(r_s_dir, l_peak_dir, s=100, alpha=0.7, color='purple')
    
    # Добавяне на етикети
    for i, (r_s, l_peak) in enumerate(zip(r_s_dir, l_peak_dir)):
        ax4.annotate(directions[i][2], (r_s, l_peak), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax4.set_xlabel('r_s [Mpc]')
    ax4.set_ylabel('l_peak')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anisotropic_effects_no_lambda.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Количествени резултати
    print(f"\n📊 АНИЗОТРОПНИ СТАТИСТИКИ:")
    print(f"{'Параметър':<15} {'Минимум':<12} {'Максимум':<12} {'Диапазон':<12} {'CV %':<10}")
    print("-" * 70)
    
    cosmo = NoLambdaCosmology(epsilon_bao=0.03, epsilon_cmb=0.02)
    
    r_s_all = [cosmo.sound_horizon_scale(theta=theta, phi=phi) for theta, phi, _ in directions]
    l_peak_all = [cosmo.cmb_peak_position(theta=theta, phi=phi) for theta, phi, _ in directions]
    
    r_s_min, r_s_max = min(r_s_all), max(r_s_all)
    l_peak_min, l_peak_max = min(l_peak_all), max(l_peak_all)
    
    r_s_cv = np.std(r_s_all) / np.mean(r_s_all) * 100
    l_peak_cv = np.std(l_peak_all) / np.mean(l_peak_all) * 100
    
    print(f"{'r_s [Mpc]':<15} {r_s_min:<12.3f} {r_s_max:<12.3f} {r_s_max-r_s_min:<12.3f} {r_s_cv:<10.2f}")
    print(f"{'l_peak':<15} {l_peak_min:<12.1f} {l_peak_max:<12.1f} {l_peak_max-l_peak_min:<12.1f} {l_peak_cv:<10.2f}")


def main():
    """Основна функция за анализ"""
    
    print("🌌 АНАЛИЗ НА BAO И CMB БЕЗ ТЪМНА ЕНЕРГИЯ")
    print("=" * 80)
    
    # Етап 1: Сравнение на основните модели
    try:
        lcdm, no_lambda, diag = compare_cosmological_models()
        print("\n✅ Сравнението на моделите завърши успешно!")
    except Exception as e:
        print(f"\n❌ Грешка в сравнението на моделите: {e}")
        return
    
    # Етап 2: BAO анализ
    try:
        analyze_bao_effects()
        print("\n✅ BAO анализът завърши успешно!")
    except Exception as e:
        print(f"\n❌ Грешка в BAO анализа: {e}")
    
    # Етап 3: CMB анализ
    try:
        analyze_cmb_effects()
        print("\n✅ CMB анализът завърши успешно!")
    except Exception as e:
        print(f"\n❌ Грешка в CMB анализа: {e}")
    
    # Етап 4: Анизотропни ефекти
    try:
        analyze_anisotropic_effects()
        print("\n✅ Анализът на анизотропните ефекти завърши успешно!")
    except Exception as e:
        print(f"\n❌ Грешка в анализа на анизотропните ефекти: {e}")
    
    print("\n🎯 ЗАКЛЮЧЕНИЯ:")
    print("-" * 20)
    print("1. Моделът без тъмна енергия показва значителни разлики от ΛCDM")
    print("2. BAO скалата r_s се променя поради липсата на ускорено разширение")
    print("3. CMB първия пик се измества поради модифицираната геометрия")
    print("4. Анизотропните ефекти въвеждат допълнителни посочни вариации")
    print("5. Необходимо е сравнение с реални наблюдателни данни")


if __name__ == "__main__":
    main() 