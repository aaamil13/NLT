#!/usr/bin/env python3
"""
Анализ на промените в космологичните изчисления при анизотропно забавяне

Този скрипт сравнява:
1. Стандартна ΛCDM космология
2. Изотропна нелинейна времева космология  
3. Анизотропна нелинейна времева космология

Показва как посочното забавяне влияе върху наблюдаемите величини.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import seaborn as sns
from typing import Dict, List, Tuple
import logging

# Импортиране на локални модули
from anisotropic_nonlinear_time import AnisotropicNonlinearTimeCosmology

# Настройка на стиловете  
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Настройка на логирането
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Физични константи
c = 299792458  # м/с


class StandardCosmology:
    """Стандартна ΛCDM космология за сравнение"""
    
    def __init__(self, H0=67.4, Omega_m=0.315, Omega_Lambda=0.685):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        
    def hubble_function(self, z):
        """Стандартна Хъбъл функция"""
        z = np.asarray(z)
        return self.H0 * np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_Lambda)
        
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


class IsotropicNonlinearTime:
    """Изотропна нелинейна времева космология"""
    
    def __init__(self, H0=67.4, Omega_m=0.315, Omega_Lambda=0.685,
                 alpha=1.5, beta=0.0, gamma=0.5, delta=0.1):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
    def time_delay(self, z):
        """Изотропно времево забавяне"""
        z = np.asarray(z)
        one_plus_z = 1 + z
        z_safe = np.maximum(z, 1e-10)
        
        return (self.alpha * z_safe**self.beta * 
                np.exp(-self.gamma * z_safe) / one_plus_z + 
                self.delta * np.log(one_plus_z))
    
    def hubble_function(self, z):
        """Модифицирана Хъбъл функция"""
        z = np.asarray(z)
        E_z = np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_Lambda)
        t_z = self.time_delay(z)
        
        return self.H0 * E_z * (1 + self.alpha * t_z)
    
    def angular_diameter_distance(self, z):
        """Модифицирано ъглово разстояние"""
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


def analyze_anisotropic_effects():
    """Анализ на анизотропните ефекти върху космологията"""
    
    print("🔬 АНАЛИЗ НА АНИЗОТРОПНИТЕ ЕФЕКТИ")
    print("=" * 60)
    
    # Създаване на космологичните модели
    lambda_cdm = StandardCosmology()
    isotropic_nl = IsotropicNonlinearTime()
    
    # Различни степени на анизотропия
    anisotropy_levels = [
        (0.0, 0.0, 0.0, "Изотропно"),
        (0.02, 0.01, 0.005, "Слаба анизотропия"),
        (0.05, 0.03, 0.02, "Умерена анизотропия"),
        (0.1, 0.06, 0.04, "Силна анизотропия")
    ]
    
    # Червени отмествания за анализ
    z_range = np.logspace(-2, 0.5, 50)  # z от 0.01 до ~3.16
    
    # Тестови посоки
    directions = [
        (0, 0, "Полярна (z-ос)"),
        (np.pi/2, 0, "Екваториална (x-ос)"),
        (np.pi/2, np.pi/2, "Екваториална (y-ос)"),
        (np.pi/4, np.pi/4, "Диагонална")
    ]
    
    # Анализ на различни степени на анизотропия
    print("\n📊 АНАЛИЗ НА РАЗЛИЧНИ СТЕПЕНИ НА АНИЗОТРОПИЯ:")
    print("-" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ефекти на анизотропното забавяне на разширението', fontsize=16)
    
    # Subplot 1: Хъбъл функция
    ax1 = axes[0, 0]
    ax1.set_title('Хъбъл параметър H(z)')
    
    # Стандартни модели
    H_lcdm = lambda_cdm.hubble_function(z_range)
    H_iso = isotropic_nl.hubble_function(z_range)
    
    ax1.plot(z_range, H_lcdm, '--', label='ΛCDM стандартен', linewidth=2, color='black')
    ax1.plot(z_range, H_iso, '-', label='Изотропно НВ', linewidth=2, color='blue')
    
    colors = ['green', 'orange', 'red', 'purple']
    
    for i, (eps_x, eps_y, eps_z, label) in enumerate(anisotropy_levels[1:]):
        aniso_cosmo = AnisotropicNonlinearTimeCosmology(
            epsilon_x=eps_x, epsilon_y=eps_y, epsilon_z=eps_z,
            tau_x=eps_x*0.8, tau_y=eps_y*0.8, tau_z=eps_z*0.8
        )
        
        # Средно по посоки
        H_avg = np.zeros_like(z_range)
        for theta, phi, _ in directions:
            H_dir = aniso_cosmo.anisotropic_hubble_function(z_range, theta, phi)
            H_avg += H_dir
        H_avg /= len(directions)
        
        ax1.plot(z_range, H_avg, '-', label=f'{label}', 
                linewidth=2, color=colors[i])
    
    ax1.set_xlabel('Червено отместване z')
    ax1.set_ylabel('H(z) [km/s/Mpc]')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Ъглово разстояние
    ax2 = axes[0, 1]
    ax2.set_title('Ъглово диаметрово разстояние')
    
    D_A_lcdm = lambda_cdm.angular_diameter_distance(z_range)
    D_A_iso = isotropic_nl.angular_diameter_distance(z_range)
    
    ax2.plot(z_range, D_A_lcdm, '--', label='ΛCDM стандартен', linewidth=2, color='black')
    ax2.plot(z_range, D_A_iso, '-', label='Изотропно НВ', linewidth=2, color='blue')
    
    for i, (eps_x, eps_y, eps_z, label) in enumerate(anisotropy_levels[1:]):
        aniso_cosmo = AnisotropicNonlinearTimeCosmology(
            epsilon_x=eps_x, epsilon_y=eps_y, epsilon_z=eps_z,
            tau_x=eps_x*0.8, tau_y=eps_y*0.8, tau_z=eps_z*0.8
        )
        
        # Средно по посоки
        D_A_avg = np.zeros_like(z_range)
        for theta, phi, _ in directions:
            D_A_dir = aniso_cosmo.anisotropic_angular_diameter_distance(z_range, theta, phi)
            D_A_avg += D_A_dir
        D_A_avg /= len(directions)
        
        ax2.plot(z_range, D_A_avg, '-', label=f'{label}', 
                linewidth=2, color=colors[i])
    
    ax2.set_xlabel('Червено отместване z')
    ax2.set_ylabel('D_A(z) [Mpc]')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Относителни отклонения от ΛCDM
    ax3 = axes[1, 0]
    ax3.set_title('Относителни отклонения от ΛCDM')
    
    for i, (eps_x, eps_y, eps_z, label) in enumerate(anisotropy_levels[1:]):
        aniso_cosmo = AnisotropicNonlinearTimeCosmology(
            epsilon_x=eps_x, epsilon_y=eps_y, epsilon_z=eps_z,
            tau_x=eps_x*0.8, tau_y=eps_y*0.8, tau_z=eps_z*0.8
        )
        
        # Средно по посоки
        H_avg = np.zeros_like(z_range)
        for theta, phi, _ in directions:
            H_dir = aniso_cosmo.anisotropic_hubble_function(z_range, theta, phi)
            H_avg += H_dir
        H_avg /= len(directions)
        
        deviation = (H_avg - H_lcdm) / H_lcdm * 100
        ax3.plot(z_range, deviation, '-', label=f'{label}', 
                linewidth=2, color=colors[i])
    
    ax3.set_xlabel('Червено отместване z')
    ax3.set_ylabel('Δh/H_ΛCDM [%]')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Subplot 4: Посочни вариации
    ax4 = axes[1, 1]
    ax4.set_title('Посочни вариации (умерена анизотропия)')
    
    aniso_cosmo = AnisotropicNonlinearTimeCosmology(
        epsilon_x=0.05, epsilon_y=0.03, epsilon_z=0.02,
        tau_x=0.04, tau_y=0.024, tau_z=0.016
    )
    
    colors_dir = ['red', 'green', 'blue', 'orange']
    
    for i, (theta, phi, dir_name) in enumerate(directions):
        H_dir = aniso_cosmo.anisotropic_hubble_function(z_range, theta, phi)
        ax4.plot(z_range, H_dir, '-', label=f'{dir_name}', 
                linewidth=2, color=colors_dir[i])
    
    ax4.plot(z_range, H_lcdm, '--', label='ΛCDM еталон', linewidth=2, color='black')
    ax4.set_xlabel('Червено отместване z')
    ax4.set_ylabel('H(z) [km/s/Mpc]')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anisotropic_effects_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Количествени резултати
    print("\n📈 КОЛИЧЕСТВЕНИ РЕЗУЛТАТИ:")
    print("-" * 40)
    
    z_test = np.array([0.1, 0.5, 1.0, 2.0])
    
    print(f"{'z':<8} {'ΛCDM':<12} {'Изотропно':<12} {'Анизотропно':<12} {'Отклонение':<12}")
    print("-" * 60)
    
    H_lcdm_test = lambda_cdm.hubble_function(z_test)
    H_iso_test = isotropic_nl.hubble_function(z_test)
    
    aniso_cosmo = AnisotropicNonlinearTimeCosmology(
        epsilon_x=0.05, epsilon_y=0.03, epsilon_z=0.02,
        tau_x=0.04, tau_y=0.024, tau_z=0.016
    )
    
    # Средно по посоки за анизотропния модел
    H_aniso_test = np.zeros_like(z_test)
    for theta, phi, _ in directions:
        H_dir = aniso_cosmo.anisotropic_hubble_function(z_test, theta, phi)
        H_aniso_test += H_dir
    H_aniso_test /= len(directions)
    
    for i, z in enumerate(z_test):
        deviation = (H_aniso_test[i] - H_lcdm_test[i]) / H_lcdm_test[i] * 100
        print(f"{z:<8.1f} {H_lcdm_test[i]:<12.1f} {H_iso_test[i]:<12.1f} {H_aniso_test[i]:<12.1f} {deviation:<12.1f}%")
    
    return aniso_cosmo


def analyze_directional_variations():
    """Анализ на посочните вариации"""
    
    print("\n🧭 АНАЛИЗ НА ПОСОЧНИТЕ ВАРИАЦИИ")
    print("=" * 60)
    
    # Създаване на силно анизотропен модел
    aniso_cosmo = AnisotropicNonlinearTimeCosmology(
        epsilon_x=0.1, epsilon_y=0.06, epsilon_z=0.04,
        tau_x=0.08, tau_y=0.048, tau_z=0.032,
        theta_preference=np.pi/3, phi_preference=np.pi/4,
        angular_strength=0.8
    )
    
    # Създаване на мрежа от посоки
    n_theta = 20
    n_phi = 40
    
    theta_grid = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2*np.pi, n_phi)
    
    # Тестово червено отместване
    z_test = 1.0
    
    # Изчисляване на H(z) за всички посоки
    H_map = np.zeros((n_theta, n_phi))
    
    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            H_map[i, j] = aniso_cosmo.anisotropic_hubble_function(z_test, theta, phi)
    
    # Създаване на небесна карта
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Небесна карта в полярни координати
    ax1 = axes[0]
    
    # Конвертиране в декартови координати за визуализация
    THETA, PHI = np.meshgrid(theta_grid, phi_grid)
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    
    im1 = ax1.contourf(PHI, THETA, H_map.T, levels=50, cmap='RdYlBu_r')
    ax1.set_xlabel('Азимутен ъгъл φ [радиани]')
    ax1.set_ylabel('Полярен ъгъл θ [радиани]')
    ax1.set_title(f'Небесна карта на H(z={z_test})')
    
    # Добавяне на цветова скала
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('H(z) [km/s/Mpc]')
    
    # Subplot 2: Статистики на вариациите
    ax2 = axes[1]
    
    H_flat = H_map.flatten()
    H_mean = np.mean(H_flat)
    H_std = np.std(H_flat)
    
    ax2.hist(H_flat, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(H_mean, color='red', linestyle='--', linewidth=2, label=f'Средно: {H_mean:.1f}')
    ax2.axvline(H_mean + H_std, color='orange', linestyle='--', linewidth=2, label=f'+1σ: {H_mean+H_std:.1f}')
    ax2.axvline(H_mean - H_std, color='orange', linestyle='--', linewidth=2, label=f'-1σ: {H_mean-H_std:.1f}')
    
    ax2.set_xlabel('H(z) [km/s/Mpc]')
    ax2.set_ylabel('Честота')
    ax2.set_title('Разпределение на H(z) по посоки')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('directional_variations_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Статистически анализ
    print(f"\n📊 СТАТИСТИКИ НА ПОСОЧНИТЕ ВАРИАЦИИ:")
    print("-" * 40)
    print(f"Средна стойност: {H_mean:.3f} km/s/Mpc")
    print(f"Стандартно отклонение: {H_std:.3f} km/s/Mpc")
    print(f"Коефициент на вариация: {H_std/H_mean*100:.2f}%")
    print(f"Минимална стойност: {np.min(H_flat):.3f} km/s/Mpc")
    print(f"Максимална стойност: {np.max(H_flat):.3f} km/s/Mpc")
    print(f"Диапазон: {np.max(H_flat) - np.min(H_flat):.3f} km/s/Mpc")
    
    # Предпочитана посока
    diagnostics = aniso_cosmo.anisotropy_diagnostics()
    print(f"\nПредпочитана посока: θ={diagnostics['theta_pref_deg']:.1f}°, φ={diagnostics['phi_pref_deg']:.1f}°")
    
    return H_map, theta_grid, phi_grid


def compare_observational_effects():
    """Сравнение на наблюдаемите ефекти"""
    
    print("\n🔭 СРАВНЕНИЕ НА НАБЛЮДАЕМИТЕ ЕФЕКТИ")
    print("=" * 60)
    
    # Създаване на моделите
    lambda_cdm = StandardCosmology()
    aniso_cosmo = AnisotropicNonlinearTimeCosmology(
        epsilon_x=0.05, epsilon_y=0.03, epsilon_z=0.02,
        tau_x=0.04, tau_y=0.024, tau_z=0.016
    )
    
    # Симулация на наблюдателни данни
    z_obs = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    
    # Различни посоки на наблюдение
    directions = [
        (0, 0, "Полярна"),
        (np.pi/2, 0, "Екваториална-X"),
        (np.pi/2, np.pi/2, "Екваториална-Y"),
        (np.pi/4, np.pi/4, "Диагонална")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Наблюдаеми ефекти на анизотропното забавяне', fontsize=16)
    
    # Subplot 1: Хъбъл диаграма
    ax1 = axes[0, 0]
    ax1.set_title('Модифицирана Хъбъл диаграма')
    
    # Стандартна ΛCDM
    H_lcdm = lambda_cdm.hubble_function(z_obs)
    ax1.plot(z_obs, H_lcdm, 'k--', linewidth=3, label='ΛCDM стандартен')
    
    # Анизотропни наблюдения
    colors = ['red', 'green', 'blue', 'orange']
    
    for i, (theta, phi, dir_name) in enumerate(directions):
        H_aniso = aniso_cosmo.anisotropic_hubble_function(z_obs, theta, phi)
        ax1.plot(z_obs, H_aniso, 'o-', color=colors[i], linewidth=2, 
                label=f'{dir_name} посока')
    
    ax1.set_xlabel('Червено отместване z')
    ax1.set_ylabel('H(z) [km/s/Mpc]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Ъглови разстояния
    ax2 = axes[0, 1]
    ax2.set_title('Ъглови диаметрови разстояния')
    
    # Стандартна ΛCDM
    D_A_lcdm = lambda_cdm.angular_diameter_distance(z_obs)
    ax2.plot(z_obs, D_A_lcdm, 'k--', linewidth=3, label='ΛCDM стандартен')
    
    # Анизотропни наблюдения
    for i, (theta, phi, dir_name) in enumerate(directions):
        D_A_aniso = aniso_cosmo.anisotropic_angular_diameter_distance(z_obs, theta, phi)
        ax2.plot(z_obs, D_A_aniso, 'o-', color=colors[i], linewidth=2, 
                label=f'{dir_name} посока')
    
    ax2.set_xlabel('Червено отместване z')
    ax2.set_ylabel('D_A(z) [Mpc]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Относителни отклонения
    ax3 = axes[1, 0]
    ax3.set_title('Относителни отклонения от ΛCDM')
    
    for i, (theta, phi, dir_name) in enumerate(directions):
        H_aniso = aniso_cosmo.anisotropic_hubble_function(z_obs, theta, phi)
        deviation = (H_aniso - H_lcdm) / H_lcdm * 100
        ax3.plot(z_obs, deviation, 'o-', color=colors[i], linewidth=2, 
                label=f'{dir_name} посока')
    
    ax3.set_xlabel('Червено отместване z')
    ax3.set_ylabel('Δн/H_ΛCDM [%]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Subplot 4: Анизотропни сигнатури
    ax4 = axes[1, 1]
    ax4.set_title('Анизотропни сигнатури')
    
    # Изчисляване на дисперсията по посоки за различни z
    z_fine = np.linspace(0.1, 2.0, 20)
    variances = []
    
    for z_val in z_fine:
        H_values = []
        for theta, phi, _ in directions:
            H_dir = aniso_cosmo.anisotropic_hubble_function(z_val, theta, phi)
            H_values.append(H_dir)
        
        variance = np.var(H_values) / np.mean(H_values)**2 * 100  # CV в %
        variances.append(variance)
    
    ax4.plot(z_fine, variances, 'ro-', linewidth=2, markersize=6)
    ax4.set_xlabel('Червено отместване z')
    ax4.set_ylabel('Коефициент на вариация [%]')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('observational_effects_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Заключения
    print("\n📝 ЗАКЛЮЧЕНИЯ:")
    print("-" * 20)
    
    max_deviation = 0
    for i, (theta, phi, dir_name) in enumerate(directions):
        H_aniso = aniso_cosmo.anisotropic_hubble_function(z_obs, theta, phi)
        deviation = np.max(np.abs((H_aniso - H_lcdm) / H_lcdm * 100))
        
        print(f"{dir_name:<20}: Максимално отклонение {deviation:.2f}%")
        max_deviation = max(max_deviation, deviation)
    
    print(f"\nОбщо максимално отклонение: {max_deviation:.2f}%")
    
    # Анизотропни диагностики
    diagnostics = aniso_cosmo.anisotropy_diagnostics()
    print(f"Обща анизотропна сила: {diagnostics['total_anisotropy']:.3f}")
    print(f"Общо времево забавяне: {diagnostics['total_delay']:.3f}")


def main():
    """Основна функция за анализ"""
    
    print("🌌 АНАЛИЗ НА АНИЗОТРОПНОТО ЗАБАВЯНЕ НА РАЗШИРЕНИЕТО")
    print("=" * 80)
    
    # Тестване на основната функционалност
    try:
        aniso_cosmo = analyze_anisotropic_effects()
        print("\n✅ Основният анализ завърши успешно!")
    except Exception as e:
        print(f"\n❌ Грешка в основния анализ: {e}")
        return
    
    # Анализ на посочните вариации
    try:
        H_map, theta_grid, phi_grid = analyze_directional_variations()
        print("\n✅ Анализът на посочните вариации завърши успешно!")
    except Exception as e:
        print(f"\n❌ Грешка в анализа на посочните вариации: {e}")
    
    # Сравнение на наблюдаемите ефекти
    try:
        compare_observational_effects()
        print("\n✅ Сравнението на наблюдаемите ефекти завърши успешно!")
    except Exception as e:
        print(f"\n❌ Грешка в сравнението на наблюдаемите ефекти: {e}")
    
    print("\n🎯 РЕЗЮМЕ:")
    print("-" * 15)
    print("Анизотропното забавяне на разширението води до:")
    print("1. Посочно зависими космологични параметри")
    print("2. Модифицирани наблюдаеми величини")
    print("3. Нови тестируеми предсказания")
    print("4. Потенциални обяснения на космологичните аномалии")
    

if __name__ == "__main__":
    main() 