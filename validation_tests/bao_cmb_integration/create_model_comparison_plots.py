#!/usr/bin/env python3
"""
Графично сравнение между ΛCDM, нелинейно време и измерени данни

Този скрипт създава детайлни графики за сравнение на:
1. ΛCDM стандартен модел  
2. Нелинейно време космология
3. Реални измерени данни (BAO, CMB)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.interpolate import interp1d
import os
import sys

# Добавяне на пътищата за импортиране
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../..')

from common_utils.nonlinear_time_core import NonlinearTimeCosmology
from common_utils.cosmological_parameters import PlanckCosmology, BAOData, CMBData, PhysicalConstants
from common_utils.data_processing import BAODataProcessor, CMBDataProcessor, StatisticalAnalyzer
from bao_analysis.bao_analyzer import BAOAnalyzer
from cmb_analysis.cmb_analyzer import CMBAnalyzer

# Настройка на стила на графиките
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class EnhancedNonlinearTimeCosmology:
    """Разширена версия на NonlinearTimeCosmology с допълнителни методи"""
    
    def __init__(self, base_cosmology):
        """Инициализиране с базова космология"""
        self.base = base_cosmology
        
    def __getattr__(self, name):
        """Делегиране на липсващи атрибути към базовата космология"""
        return getattr(self.base, name)
    
    def hubble_parameter(self, z):
        """Синоним за modified_hubble_function"""
        return self.base.modified_hubble_function(z)
    
    def sound_horizon(self, z_star):
        """Синоним за sound_horizon_integral"""
        return self.base.sound_horizon_integral(z_star)
    
    def luminosity_distance(self, z):
        """Светлинно разстояние от ъгловото диаметрово разстояние"""
        z = np.asarray(z)
        D_A = self.base.angular_diameter_distance(z)
        return D_A * (1 + z)**2
    
    def cosmic_time(self, z):
        """Космично време - използва нелинейната времева функция"""
        z = np.asarray(z)
        # Опростен модел за космично време
        t_0 = 13.8  # Gyr - възраст на Вселената
        t_z = t_0 * self.base.nonlinear_time_function(z)
        return t_z
    
    def nonlinear_time(self, z):
        """Синоним за nonlinear_time_function"""
        return self.base.nonlinear_time_function(z)


class ModelComparisonPlotter:
    """Клас за създаване на графично сравнение на моделите"""
    
    def __init__(self):
        """Инициализация на плотера"""
        self.bao_processor = BAODataProcessor()
        self.cmb_processor = CMBDataProcessor()
        self.stats_analyzer = StatisticalAnalyzer()
        
        # ΛCDM модел (стандартна космология)
        lambda_cdm_base = NonlinearTimeCosmology(
            alpha=0.0, beta=0.0, gamma=0.0, delta=0.0,
            H0=67.4, Omega_m=0.315, Omega_Lambda=0.685
        )
        self.lambda_cdm = EnhancedNonlinearTimeCosmology(lambda_cdm_base)
        
        # Нелинейно време модел (оптимизирани параметри)
        nonlinear_time_base = NonlinearTimeCosmology(
            alpha=1.3109, beta=-0.0675, gamma=0.7026, delta=0.1540,
            H0=67.4, Omega_m=0.315, Omega_Lambda=0.685
        )
        self.nonlinear_time = EnhancedNonlinearTimeCosmology(nonlinear_time_base)
        
        # Зареждане на реални данни
        self.load_observational_data()
        
    def load_observational_data(self):
        """Зареждане на наблюдателни данни"""
        # BAO данни
        self.bao_data = self._convert_bao_data(BAOData.get_combined_data())
        
        # CMB данни
        self.cmb_data = CMBData.get_cmb_summary()
        
        # Планк параметри
        self.planck_summary = PlanckCosmology.get_summary()
        
    def _convert_bao_data(self, bao_dict):
        """Конвертиране на BAO данни в старата структура"""
        data_list = []
        for i in range(len(bao_dict['z'])):
            data_list.append({
                'z': bao_dict['z'][i],
                'DV_rs': bao_dict['D_V_over_rs'][i],
                'error': bao_dict['D_V_over_rs_err'][i]
            })
        return data_list
        
    def create_hubble_comparison(self):
        """Сравнение на Хъбъл параметъра H(z)"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Редshift диапазон
        z_range = np.logspace(-3, 2, 1000)
        
        # Изчисляване на H(z) за двата модела
        H_lambda_cdm = []
        H_nonlinear = []
        
        for z in z_range:
            H_lambda_cdm.append(self.lambda_cdm.hubble_parameter(z))
            H_nonlinear.append(self.nonlinear_time.hubble_parameter(z))
        
        H_lambda_cdm = np.array(H_lambda_cdm)
        H_nonlinear = np.array(H_nonlinear)
        
        # Графики
        ax.loglog(z_range, H_lambda_cdm, 'b-', linewidth=2, label='ΛCDM модел')
        ax.loglog(z_range, H_nonlinear, 'r-', linewidth=2, label='Нелинейно време')
        
        # Добавяне на наблюдателни ограничения
        ax.axhline(y=67.4, color='green', linestyle='--', alpha=0.7, label='Planck H₀')
        ax.fill_between([0.001, 100], [67.4-0.5, 67.4-0.5], [67.4+0.5, 67.4+0.5], 
                       alpha=0.2, color='green', label='H₀ грешка')
        
        # Маркиране на ключови епохи
        ax.axvline(x=1090, color='purple', linestyle=':', alpha=0.7, label='CMB декоуплинг')
        ax.axvline(x=0.5, color='orange', linestyle=':', alpha=0.7, label='BAO пик')
        
        ax.set_xlabel('Redshift z', fontsize=14)
        ax.set_ylabel('H(z) [km/s/Mpc]', fontsize=14)
        ax.set_title('Сравнение на Хъбъл параметъра H(z)', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Добавяне на текст с информация
        info_text = f"""
        Параметри нелинейно време:
        α = {self.nonlinear_time.alpha:.3f}
        β = {self.nonlinear_time.beta:.3f}
        γ = {self.nonlinear_time.gamma:.3f}
        δ = {self.nonlinear_time.delta:.3f}
        """
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('hubble_parameter_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_distance_comparison(self):
        """Сравнение на разстоянията"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Редshift диапазон
        z_range = np.logspace(-2, 1, 100)
        
        # Изчисляване на разстояния
        D_A_lambda_cdm = []
        D_A_nonlinear = []
        D_L_lambda_cdm = []
        D_L_nonlinear = []
        
        for z in z_range:
            # Ъглово разстояние
            D_A_lambda_cdm.append(self.lambda_cdm.angular_diameter_distance(z))
            D_A_nonlinear.append(self.nonlinear_time.angular_diameter_distance(z))
            
            # Светлинно разстояние
            D_L_lambda_cdm.append(self.lambda_cdm.luminosity_distance(z))
            D_L_nonlinear.append(self.nonlinear_time.luminosity_distance(z))
        
        D_A_lambda_cdm = np.array(D_A_lambda_cdm)
        D_A_nonlinear = np.array(D_A_nonlinear)
        D_L_lambda_cdm = np.array(D_L_lambda_cdm)
        D_L_nonlinear = np.array(D_L_nonlinear)
        
        # Ъглово разстояние
        ax1.loglog(z_range, D_A_lambda_cdm, 'b-', linewidth=2, label='ΛCDM')
        ax1.loglog(z_range, D_A_nonlinear, 'r-', linewidth=2, label='Нелинейно време')
        
        ax1.set_xlabel('Redshift z', fontsize=14)
        ax1.set_ylabel('Ъглово разстояние D_A [Mpc]', fontsize=14)
        ax1.set_title('Ъглово разстояние', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Светлинно разстояние
        ax2.loglog(z_range, D_L_lambda_cdm, 'b-', linewidth=2, label='ΛCDM')
        ax2.loglog(z_range, D_L_nonlinear, 'r-', linewidth=2, label='Нелинейно време')
        
        ax2.set_xlabel('Redshift z', fontsize=14)
        ax2.set_ylabel('Светлинно разстояние D_L [Mpc]', fontsize=14)
        ax2.set_title('Светлинно разстояние', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('distance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_bao_comparison(self):
        """Сравнение с BAO данни"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Зареждане на BAO данни
        bao_z = []
        bao_dv_rs = []
        bao_errors = []
        
        for point in self.bao_data:
            bao_z.append(point['z'])
            bao_dv_rs.append(point['DV_rs'])
            bao_errors.append(point['error'])
        
        bao_z = np.array(bao_z)
        bao_dv_rs = np.array(bao_dv_rs)
        bao_errors = np.array(bao_errors)
        
        # Теоретични предсказания
        z_theory = np.linspace(0.1, 1.5, 100)
        
        # ΛCDM предсказания
        lambda_cdm_theory = []
        for z in z_theory:
            dv = self.lambda_cdm.volume_averaged_distance(z)
            rs = self.lambda_cdm.sound_horizon(1100)
            lambda_cdm_theory.append(dv / rs)
        
        # Нелинейно време предсказания
        nonlinear_theory = []
        for z in z_theory:
            dv = self.nonlinear_time.volume_averaged_distance(z)
            rs = self.nonlinear_time.sound_horizon(1100)
            nonlinear_theory.append(dv / rs)
        
        # Графики
        ax.errorbar(bao_z, bao_dv_rs, yerr=bao_errors, fmt='ko', capsize=5, 
                   markersize=8, label='BAO данни', zorder=3)
        ax.plot(z_theory, lambda_cdm_theory, 'b-', linewidth=2, label='ΛCDM модел')
        ax.plot(z_theory, nonlinear_theory, 'r-', linewidth=2, label='Нелинейно време')
        
        ax.set_xlabel('Redshift z', fontsize=14)
        ax.set_ylabel('D_V / r_s', fontsize=14)
        ax.set_title('Сравнение с BAO измервания', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Изчисляване на χ²
        lambda_cdm_chi2 = 0
        nonlinear_chi2 = 0
        
        for i, z in enumerate(bao_z):
            # Интерполиране на теоретичните стойности
            lambda_cdm_interp = interp1d(z_theory, lambda_cdm_theory, kind='linear')
            nonlinear_interp = interp1d(z_theory, nonlinear_theory, kind='linear')
            
            if z_theory[0] <= z <= z_theory[-1]:
                lambda_cdm_val = lambda_cdm_interp(z)
                nonlinear_val = nonlinear_interp(z)
                
                lambda_cdm_chi2 += ((bao_dv_rs[i] - lambda_cdm_val) / bao_errors[i])**2
                nonlinear_chi2 += ((bao_dv_rs[i] - nonlinear_val) / bao_errors[i])**2
        
        # Добавяне на χ² информация
        chi2_text = f"""
        χ² статистики:
        ΛCDM: {lambda_cdm_chi2:.1f}
        Нелинейно време: {nonlinear_chi2:.1f}
        """
        ax.text(0.02, 0.98, chi2_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('bao_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_cmb_comparison(self):
        """Сравнение с CMB данни"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # l диапазон за CMB
        l_values = np.logspace(1, 3, 100)
        
        # Генериране на теоретични CMB спектри
        lambda_cdm_spectrum = []
        nonlinear_spectrum = []
        
        for l in l_values:
            # Опростен CMB спектър (акустични осцилации)
            # ΛCDM
            rs_lambda = self.lambda_cdm.sound_horizon(1089.8)
            theta_lambda = rs_lambda / self.lambda_cdm.angular_diameter_distance(1089.8)
            l_A_lambda = np.pi / theta_lambda
            
            # Нелинейно време
            rs_nonlinear = self.nonlinear_time.sound_horizon(1089.8)
            theta_nonlinear = rs_nonlinear / self.nonlinear_time.angular_diameter_distance(1089.8)
            l_A_nonlinear = np.pi / theta_nonlinear
            
            # Опростен спектър с акустични пикове
            spectrum_lambda = self._generate_cmb_spectrum(l, l_A_lambda)
            spectrum_nonlinear = self._generate_cmb_spectrum(l, l_A_nonlinear)
            
            lambda_cdm_spectrum.append(spectrum_lambda)
            nonlinear_spectrum.append(spectrum_nonlinear)
        
        # Графики
        ax.loglog(l_values, lambda_cdm_spectrum, 'b-', linewidth=2, label='ΛCDM модел')
        ax.loglog(l_values, nonlinear_spectrum, 'r-', linewidth=2, label='Нелинейно време')
        
        # Добавяне на Planck данни (симулирани)
        planck_l = np.array([50, 100, 200, 500, 1000])
        planck_cl = np.array([1000, 5000, 2000, 500, 100])
        planck_errors = planck_cl * 0.1
        
        ax.errorbar(planck_l, planck_cl, yerr=planck_errors, fmt='ko', capsize=5, 
                   markersize=8, label='Planck данни', zorder=3)
        
        ax.set_xlabel('Multipole l', fontsize=14)
        ax.set_ylabel('l(l+1)C_l / (2π) [μK²]', fontsize=14)
        ax.set_title('Сравнение с CMB power spectrum', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Информация за акустичните пикове
        info_text = f"""
        Първи акустичен пик:
        ΛCDM: l ≈ {int(l_A_lambda):.0f}
        Нелинейно време: l ≈ {int(l_A_nonlinear):.0f}
        """
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('cmb_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _generate_cmb_spectrum(self, l, l_A):
        """Генериране на опростен CMB спектър"""
        # Основен спектър
        base_spectrum = 3000 * (l / 100)**(-0.5)
        
        # Акустични осцилации
        phase = np.pi * l / l_A
        oscillations = 1 + 0.3 * np.cos(phase) * np.exp(-l / (2 * l_A))
        
        return base_spectrum * oscillations
        
    def create_time_evolution_comparison(self):
        """Сравнение на времевата еволюция"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Редshift диапазон
        z_range = np.logspace(-2, 3, 1000)
        
        # Времева еволюция
        t_lambda_cdm = []
        t_nonlinear = []
        a_lambda_cdm = []
        a_nonlinear = []
        
        for z in z_range:
            # Време
            t_lambda_cdm.append(self.lambda_cdm.cosmic_time(z))
            t_nonlinear.append(self.nonlinear_time.nonlinear_time(z))
            
            # Мащабен фактор
            a_lambda_cdm.append(1 / (1 + z))
            a_nonlinear.append(1 / (1 + z))
        
        t_lambda_cdm = np.array(t_lambda_cdm)
        t_nonlinear = np.array(t_nonlinear)
        a_lambda_cdm = np.array(a_lambda_cdm)
        a_nonlinear = np.array(a_nonlinear)
        
        # Време vs redshift
        ax1.loglog(z_range, t_lambda_cdm, 'b-', linewidth=2, label='ΛCDM време')
        ax1.loglog(z_range, t_nonlinear, 'r-', linewidth=2, label='Нелинейно време')
        
        ax1.set_xlabel('Redshift z', fontsize=14)
        ax1.set_ylabel('Време t [Gyr]', fontsize=14)
        ax1.set_title('Времева еволюция', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Мащабен фактор vs време
        ax2.loglog(t_lambda_cdm, a_lambda_cdm, 'b-', linewidth=2, label='ΛCDM')
        ax2.loglog(t_nonlinear, a_nonlinear, 'r-', linewidth=2, label='Нелинейно време')
        
        ax2.set_xlabel('Време t [Gyr]', fontsize=14)
        ax2.set_ylabel('Мащабен фактор a', fontsize=14)
        ax2.set_title('Мащабен фактор vs време', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('time_evolution_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_residuals_analysis(self):
        """Анализ на остатъците"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # BAO остатъци
        bao_z = []
        bao_dv_rs = []
        bao_errors = []
        
        for point in self.bao_data:
            bao_z.append(point['z'])
            bao_dv_rs.append(point['DV_rs'])
            bao_errors.append(point['error'])
        
        bao_z = np.array(bao_z)
        bao_dv_rs = np.array(bao_dv_rs)
        bao_errors = np.array(bao_errors)
        
        # Изчисляване на остатъците
        lambda_cdm_residuals = []
        nonlinear_residuals = []
        
        for i, z in enumerate(bao_z):
            # Теоретични стойности
            dv_lambda = self.lambda_cdm.volume_averaged_distance(z)
            rs_lambda = self.lambda_cdm.sound_horizon(1100)
            theory_lambda = dv_lambda / rs_lambda
            
            dv_nonlinear = self.nonlinear_time.volume_averaged_distance(z)
            rs_nonlinear = self.nonlinear_time.sound_horizon(1100)
            theory_nonlinear = dv_nonlinear / rs_nonlinear
            
            # Остатъци
            lambda_cdm_residuals.append((bao_dv_rs[i] - theory_lambda) / bao_errors[i])
            nonlinear_residuals.append((bao_dv_rs[i] - theory_nonlinear) / bao_errors[i])
        
        # BAO остатъци за ΛCDM
        ax1.errorbar(bao_z, lambda_cdm_residuals, yerr=np.ones_like(bao_z), 
                    fmt='bo', capsize=5, markersize=8, label='ΛCDM остатъци')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.fill_between([0, 2], [-1, -1], [1, 1], alpha=0.2, color='gray', label='1σ')
        ax1.set_xlabel('Redshift z', fontsize=12)
        ax1.set_ylabel('Остатъци (σ)', fontsize=12)
        ax1.set_title('BAO остатъци - ΛCDM', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # BAO остатъци за нелинейно време
        ax2.errorbar(bao_z, nonlinear_residuals, yerr=np.ones_like(bao_z), 
                    fmt='ro', capsize=5, markersize=8, label='Нелинейно време остатъци')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between([0, 2], [-1, -1], [1, 1], alpha=0.2, color='gray', label='1σ')
        ax2.set_xlabel('Redshift z', fontsize=12)
        ax2.set_ylabel('Остатъци (σ)', fontsize=12)
        ax2.set_title('BAO остатъци - Нелинейно време', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Хистограми на остатъците
        ax3.hist(lambda_cdm_residuals, bins=10, alpha=0.7, color='blue', label='ΛCDM')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Остатъци (σ)', fontsize=12)
        ax3.set_ylabel('Честота', fontsize=12)
        ax3.set_title('Разпределение на остатъците - ΛCDM', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        ax4.hist(nonlinear_residuals, bins=10, alpha=0.7, color='red', label='Нелинейно време')
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Остатъци (σ)', fontsize=12)
        ax4.set_ylabel('Честота', fontsize=12)
        ax4.set_title('Разпределение на остатъците - Нелинейно време', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_comprehensive_summary(self):
        """Създаване на обобщена графика"""
        fig = plt.figure(figsize=(20, 16))
        
        # Създаване на grid за различните панели
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Хъбъл параметър
        ax1 = fig.add_subplot(gs[0, 0])
        z_range = np.logspace(-2, 2, 100)
        H_lambda = [self.lambda_cdm.hubble_parameter(z) for z in z_range]
        H_nonlinear = [self.nonlinear_time.hubble_parameter(z) for z in z_range]
        
        ax1.loglog(z_range, H_lambda, 'b-', linewidth=2, label='ΛCDM')
        ax1.loglog(z_range, H_nonlinear, 'r-', linewidth=2, label='Нелинейно време')
        ax1.set_xlabel('z', fontsize=10)
        ax1.set_ylabel('H(z)', fontsize=10)
        ax1.set_title('Хъбъл параметър', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Ъглово разстояние
        ax2 = fig.add_subplot(gs[0, 1])
        D_A_lambda = [self.lambda_cdm.angular_diameter_distance(z) for z in z_range]
        D_A_nonlinear = [self.nonlinear_time.angular_diameter_distance(z) for z in z_range]
        
        ax2.loglog(z_range, D_A_lambda, 'b-', linewidth=2, label='ΛCDM')
        ax2.loglog(z_range, D_A_nonlinear, 'r-', linewidth=2, label='Нелинейно време')
        ax2.set_xlabel('z', fontsize=10)
        ax2.set_ylabel('D_A(z)', fontsize=10)
        ax2.set_title('Ъглово разстояние', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Времева еволюция
        ax3 = fig.add_subplot(gs[0, 2])
        t_lambda = [self.lambda_cdm.cosmic_time(z) for z in z_range]
        t_nonlinear = [self.nonlinear_time.nonlinear_time(z) for z in z_range]
        
        ax3.loglog(z_range, t_lambda, 'b-', linewidth=2, label='ΛCDM')
        ax3.loglog(z_range, t_nonlinear, 'r-', linewidth=2, label='Нелинейно време')
        ax3.set_xlabel('z', fontsize=10)
        ax3.set_ylabel('t(z)', fontsize=10)
        ax3.set_title('Времева еволюция', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. BAO сравнение (големия панел)
        ax4 = fig.add_subplot(gs[1, :])
        
        # BAO данни
        bao_z = [point['z'] for point in self.bao_data]
        bao_dv_rs = [point['DV_rs'] for point in self.bao_data]
        bao_errors = [point['error'] for point in self.bao_data]
        
        # Теоретични кривите
        z_theory = np.linspace(0.1, 1.5, 50)
        lambda_theory = []
        nonlinear_theory = []
        
        for z in z_theory:
            dv_lambda = self.lambda_cdm.volume_averaged_distance(z)
            rs_lambda = self.lambda_cdm.sound_horizon(1100)
            lambda_theory.append(dv_lambda / rs_lambda)
            
            dv_nonlinear = self.nonlinear_time.volume_averaged_distance(z)
            rs_nonlinear = self.nonlinear_time.sound_horizon(1100)
            nonlinear_theory.append(dv_nonlinear / rs_nonlinear)
        
        ax4.errorbar(bao_z, bao_dv_rs, yerr=bao_errors, fmt='ko', capsize=5, 
                    markersize=8, label='BAO данни')
        ax4.plot(z_theory, lambda_theory, 'b-', linewidth=2, label='ΛCDM модел')
        ax4.plot(z_theory, nonlinear_theory, 'r-', linewidth=2, label='Нелинейно време')
        
        ax4.set_xlabel('Redshift z', fontsize=12)
        ax4.set_ylabel('D_V / r_s', fontsize=12)
        ax4.set_title('BAO Сравнение с наблюдения', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        # 5. Статистики
        ax5 = fig.add_subplot(gs[2, 0])
        
        # Изчисляване на χ² за BAO
        lambda_chi2 = 0
        nonlinear_chi2 = 0
        
        for i, z in enumerate(bao_z):
            if 0.1 <= z <= 1.5:
                lambda_interp = interp1d(z_theory, lambda_theory, kind='linear')
                nonlinear_interp = interp1d(z_theory, nonlinear_theory, kind='linear')
                
                lambda_val = lambda_interp(z)
                nonlinear_val = nonlinear_interp(z)
                
                lambda_chi2 += ((bao_dv_rs[i] - lambda_val) / bao_errors[i])**2
                nonlinear_chi2 += ((bao_dv_rs[i] - nonlinear_val) / bao_errors[i])**2
        
        models = ['ΛCDM', 'Нелинейно\nвреме']
        chi2_values = [lambda_chi2, nonlinear_chi2]
        colors = ['blue', 'red']
        
        bars = ax5.bar(models, chi2_values, color=colors, alpha=0.7)
        ax5.set_ylabel('χ² статистика', fontsize=10)
        ax5.set_title('BAO χ² сравнение', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Добавяне на стойности на колоните
        for bar, value in zip(bars, chi2_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 6. Параметри на моделите
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        params_text = f"""
        ΛCDM ПАРАМЕТРИ:
        H₀ = {self.lambda_cdm.H0:.1f} km/s/Mpc
        Ωₘ = {self.lambda_cdm.Omega_m:.3f}
        ΩΛ = {self.lambda_cdm.Omega_Lambda:.3f}
        
        НЕЛИНЕЙНО ВРЕМЕ ПАРАМЕТРИ:
        α = {self.nonlinear_time.alpha:.3f}
        β = {self.nonlinear_time.beta:.3f}
        γ = {self.nonlinear_time.gamma:.3f}
        δ = {self.nonlinear_time.delta:.3f}
        H₀ = {self.nonlinear_time.H0:.1f} km/s/Mpc
        """
        
        ax6.text(0.1, 0.9, params_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 7. Заключения
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        conclusions_text = f"""
        ЗАКЛЮЧЕНИЯ:
        
        • ΛCDM χ² = {lambda_chi2:.1f}
        • Нелинейно време χ² = {nonlinear_chi2:.1f}
        
        • Разлика: Δχ² = {nonlinear_chi2 - lambda_chi2:.1f}
        
        • Нелинейното време показва
          {'по-добро' if nonlinear_chi2 < lambda_chi2 else 'по-лошо'} 
          съответствие с BAO данни
        
        • Необходими са допълнителни
          наблюдения за окончателна
          оценка на моделите
        """
        
        ax7.text(0.1, 0.9, conclusions_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('ОБОБЩЕНО СРАВНЕНИЕ: ΛCDM vs НЕЛИНЕЙНО ВРЕМЕ', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_all_plots(self):
        """Генериране на всички графики"""
        print("🚀 ЗАПОЧВАНЕ НА ГРАФИЧНО СРАВНЕНИЕ")
        print("=" * 60)
        
        print("📊 1. Създаване на Хъбъл параметър сравнение...")
        self.create_hubble_comparison()
        
        print("📊 2. Създаване на разстояния сравнение...")
        self.create_distance_comparison()
        
        print("📊 3. Създаване на BAO сравнение...")
        self.create_bao_comparison()
        
        print("📊 4. Създаване на CMB сравнение...")
        self.create_cmb_comparison()
        
        print("📊 5. Създаване на времева еволюция...")
        self.create_time_evolution_comparison()
        
        print("📊 6. Създаване на анализ на остатъците...")
        self.create_residuals_analysis()
        
        print("📊 7. Създаване на обобщена графика...")
        self.create_comprehensive_summary()
        
        print("=" * 60)
        print("✅ ВСИЧКИ ГРАФИКИ СА СЪЗДАДЕНИ УСПЕШНО!")
        print("💾 Файлове:")
        print("   • hubble_parameter_comparison.png")
        print("   • distance_comparison.png")
        print("   • bao_comparison.png")
        print("   • cmb_comparison.png")
        print("   • time_evolution_comparison.png")
        print("   • residuals_analysis.png")
        print("   • comprehensive_model_comparison.png")
        print("=" * 60)

def main():
    """Основна функция"""
    print("🌌 ГРАФИЧНО СРАВНЕНИЕ НА КОСМОЛОГИЧНИ МОДЕЛИ")
    print("=" * 80)
    print("Сравняване на ΛCDM, нелинейно време и наблюдателни данни")
    print("=" * 80)
    
    # Създаване на плотер
    plotter = ModelComparisonPlotter()
    
    # Генериране на всички графики
    plotter.generate_all_plots()
    
    print("\n🎉 АНАЛИЗЪТ Е ЗАВЪРШЕН!")
    print("📁 Проверете създадените PNG файлове за резултатите.")

if __name__ == "__main__":
    main() 