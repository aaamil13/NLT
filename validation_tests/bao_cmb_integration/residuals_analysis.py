#!/usr/bin/env python3
"""
Residuals Analysis
==================

Анализ на остатъци за χ² = 223.749

Цел:
1. Идентифициране на проблемни области
2. Анализ на BAO и CMB остатъци
3. Предложения за подобрения
4. Статистически тестове
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Наши модули
from observational_data import BAOObservationalData, CMBObservationalData, LikelihoodFunctions
from no_lambda_cosmology import NoLambdaCosmology
from nested_sampling_analysis import OptimizedNestedSampling

class ResidualsAnalyzer:
    """Анализатор на остатъци за No-Lambda модел"""
    
    def __init__(self):
        """Инициализация на анализатора"""
        print("🔍 Инициализиране на Residuals Analyzer")
        
        # Заредяване на данни
        self.bao_data = BAOObservationalData()
        self.cmb_data = CMBObservationalData()
        
        # Best-fit параметри от nested sampling
        self.best_fit_params = {
            'H0': 68.4557,
            'Omega_m': 0.2576,
            'epsilon_bao': 0.0492,
            'epsilon_cmb': 0.0225
        }
        
        print(f"📊 Best-fit параметри: {self.best_fit_params}")
        
        # Инициализация на модел
        self.cosmology = NoLambdaCosmology(
            H0=self.best_fit_params['H0'],
            Omega_m=self.best_fit_params['Omega_m'],
            epsilon_bao=self.best_fit_params['epsilon_bao'],
            epsilon_cmb=self.best_fit_params['epsilon_cmb']
        )
        
        # Likelihood функция
        self.likelihood = LikelihoodFunctions(self.bao_data, self.cmb_data)
        
        print("✅ Residuals Analyzer инициализиран")
    
    def analyze_bao_residuals(self) -> Dict:
        """Анализ на BAO остатъци"""
        print("\n🔍 АНАЛИЗ НА BAO ОСТАТЪЦИ")
        print("=" * 40)
        
        # Получаване на BAO данни
        bao_obs = self.bao_data.get_combined_data()
        
        # Данните са в combined формат
        z = bao_obs['redshifts']
        DV_rs_obs = bao_obs['DV_rs']
        DV_rs_err = bao_obs['DV_rs_err']
        
        print(f"\n📊 Общо BAO точки: {len(z)}")
        
        # Теоретични стойности
        DV_rs_theory = []
        for zi in z:
            # Изчисляване на D_V/r_s
            D_A = self.cosmology.angular_diameter_distance(zi)
            H_z = self.cosmology.hubble_function(zi)
            D_H = 299792.458 / H_z  # Hubble distance
            D_V = (zi * D_A**2 * D_H)**(1/3)  # Dilation scale
            r_s = self.cosmology.sound_horizon_scale()
            DV_rs_theory.append(D_V / r_s)
        
        DV_rs_theory = np.array(DV_rs_theory)
        
        # Остатъци
        residuals = DV_rs_obs - DV_rs_theory
        normalized_residuals = residuals / DV_rs_err
        
        # Записване на данните
        predictions = DV_rs_theory
        observed = DV_rs_obs
        errors = DV_rs_err
        redshifts = z
        
        # Детайлен изход
        for i, zi in enumerate(z):
            print(f"  z={zi:.3f}: obs={DV_rs_obs[i]:.3f}, theory={DV_rs_theory[i]:.3f}, "
                  f"residual={residuals[i]:.3f} ({residuals[i]/DV_rs_obs[i]*100:.1f}%)")
        
        # Статистики
        chi2_individual = normalized_residuals**2
        
        print(f"\n📊 BAO СТАТИСТИКИ:")
        print(f"Брой точки: {len(residuals)}")
        print(f"Средни остатъци: {np.mean(residuals):.3f}")
        print(f"Стандартно отклонение: {np.std(residuals):.3f}")
        print(f"Максимален остатък: {np.max(np.abs(residuals)):.3f}")
        print(f"BAO χ²: {np.sum(chi2_individual):.3f}")
        
        # Най-проблемни точки
        problematic_indices = np.argsort(chi2_individual)[-3:]  # Топ 3 най-лоши
        print(f"\n🚨 НАЙ-ПРОБЛЕМНИ ТОЧКИ:")
        for i in problematic_indices:
            print(f"  z={redshifts[i]:.3f}: χ²={chi2_individual[i]:.1f}, "
                  f"residual={residuals[i]:.3f} ({residuals[i]/observed[i]*100:.1f}%)")
        
        return {
            'redshifts': redshifts,
            'observed': observed,
            'predicted': predictions,
            'errors': errors,
            'residuals': residuals,
            'normalized_residuals': normalized_residuals,
            'chi2_individual': chi2_individual,
            'chi2_total': np.sum(chi2_individual),
            'problematic_indices': problematic_indices
        }
    
    def analyze_cmb_residuals(self) -> Dict:
        """Анализ на CMB остатъци"""
        print("\n🔍 АНАЛИЗ НА CMB ОСТАТЪЦИ")
        print("=" * 40)
        
        # Получаване на CMB данни
        cmb_obs = self.cmb_data.get_acoustic_scale()
        
        # Теоретични предсказания
        theta_s_theory = self.cosmology.cmb_angular_scale()
        theta_s_obs = cmb_obs['theta_s']
        theta_s_err = cmb_obs['theta_s_err']
        
        # Остатъци
        residual = theta_s_obs - theta_s_theory
        normalized_residual = residual / theta_s_err
        chi2_cmb = normalized_residual**2
        
        print(f"📊 CMB СТАТИСТИКИ:")
        print(f"Наблюдавано theta_s: {theta_s_obs:.6f}")
        print(f"Теоретично theta_s: {theta_s_theory:.6f}")
        print(f"Остатък: {residual:.6f}")
        print(f"Относителен остатък: {residual/theta_s_obs*100:.1f}%")
        print(f"Нормализиран остатък: {normalized_residual:.1f}")
        print(f"CMB χ²: {chi2_cmb:.1f}")
        
        # Интерпретация
        if abs(normalized_residual) > 3:
            print("🚨 КРИТИЧНО: >3σ отклонение!")
        elif abs(normalized_residual) > 2:
            print("⚠️ ВНИМАНИЕ: >2σ отклонение")
        else:
            print("✅ Приемливо отклонение")
        
        return {
            'theta_s_obs': theta_s_obs,
            'theta_s_theory': theta_s_theory,
            'theta_s_err': theta_s_err,
            'residual': residual,
            'normalized_residual': normalized_residual,
            'chi2_cmb': chi2_cmb
        }
    
    def statistical_tests(self, bao_results: Dict, cmb_results: Dict) -> Dict:
        """Статистически тестове на остатъци"""
        print("\n🔍 СТАТИСТИЧЕСКИ ТЕСТОВЕ")
        print("=" * 40)
        
        # BAO тестове
        bao_residuals = bao_results['normalized_residuals']
        
        # Нормално разпределение тест
        shapiro_stat, shapiro_p = stats.shapiro(bao_residuals)
        print(f"📊 Shapiro-Wilk тест (нормалност):")
        print(f"  Статистика: {shapiro_stat:.3f}")
        print(f"  P-value: {shapiro_p:.6f}")
        print(f"  Резултат: {'Нормално' if shapiro_p > 0.05 else 'НЕ нормално'}")
        
        # Autocorrelation тест
        # Сортиране по z за автокорелация
        sorted_indices = np.argsort(bao_results['redshifts'])
        sorted_residuals = bao_residuals[sorted_indices]
        
        # Durbin-Watson тест
        def durbin_watson(residuals):
            diff = np.diff(residuals)
            return np.sum(diff**2) / np.sum(residuals**2)
        
        dw_stat = durbin_watson(sorted_residuals)
        print(f"\n📊 Durbin-Watson тест (автокорелация):")
        print(f"  Статистика: {dw_stat:.3f}")
        print(f"  Очаквана стойност: ~2.0")
        print(f"  Резултат: {'Добре' if 1.5 < dw_stat < 2.5 else 'Проблематично'}")
        
        # Runs тест за случайност
        median_residual = np.median(bao_residuals)
        runs, n1, n2 = 0, 0, 0
        for i, res in enumerate(bao_residuals):
            if res > median_residual:
                n1 += 1
                if i == 0 or bao_residuals[i-1] <= median_residual:
                    runs += 1
            else:
                n2 += 1
                if i == 0 or bao_residuals[i-1] > median_residual:
                    runs += 1
        
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        print(f"\n📊 Runs тест (случайност):")
        print(f"  Runs: {runs}")
        print(f"  Очаквани runs: {expected_runs:.1f}")
        print(f"  Резултат: {'Случайно' if abs(runs - expected_runs) < 2 else 'Неслучайно'}")
        
        return {
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'durbin_watson': dw_stat,
            'runs': runs,
            'expected_runs': expected_runs
        }
    
    def identify_problems(self, bao_results: Dict, cmb_results: Dict) -> List[str]:
        """Идентифициране на проблеми"""
        print("\n🔍 ИДЕНТИФИЦИРАНЕ НА ПРОБЛЕМИ")
        print("=" * 40)
        
        problems = []
        
        # BAO проблеми
        if bao_results['chi2_total'] > 20:  # Очакваме ~10 за 10 точки
            problems.append(f"BAO χ² = {bao_results['chi2_total']:.1f} >> 10 (очаквано)")
        
        # Систематични отклонения
        mean_residual = np.mean(bao_results['residuals'])
        if abs(mean_residual) > 0.5:
            problems.append(f"Систематично отклонение в BAO: {mean_residual:.3f}")
        
        # Големи индивидуални отклонения  
        max_chi2 = np.max(bao_results['chi2_individual'])
        if max_chi2 > 10:
            problems.append(f"Огромно индивидуално BAO χ² = {max_chi2:.1f}")
        
        # CMB проблеми
        if abs(cmb_results['normalized_residual']) > 2:
            problems.append(f"CMB отклонение: {cmb_results['normalized_residual']:.1f}σ")
        
        # Redshift зависимост
        z_residuals = []
        for i, z in enumerate(bao_results['redshifts']):
            z_residuals.append((z, bao_results['normalized_residuals'][i]))
        
        z_residuals.sort()
        low_z_residuals = [res for z, res in z_residuals if z < 0.5]
        high_z_residuals = [res for z, res in z_residuals if z > 1.0]
        
        if len(low_z_residuals) > 0 and len(high_z_residuals) > 0:
            low_z_mean = np.mean(low_z_residuals)
            high_z_mean = np.mean(high_z_residuals)
            if abs(low_z_mean - high_z_mean) > 1:
                problems.append(f"Redshift зависимост: low-z={low_z_mean:.1f}, high-z={high_z_mean:.1f}")
        
        print(f"🚨 НАМЕРЕНИ ПРОБЛЕМИ ({len(problems)}):")
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem}")
        
        return problems
    
    def suggest_improvements(self, problems: List[str]) -> List[str]:
        """Предложения за подобрения"""
        print("\n💡 ПРЕДЛОЖЕНИЯ ЗА ПОДОБРЕНИЯ")
        print("=" * 40)
        
        suggestions = []
        
        # Анализ на проблемите
        if any("BAO χ²" in p for p in problems):
            suggestions.append("Преразглеждане на BAO likelihood функция")
            suggestions.append("Проверка на BAO данни за outliers")
            suggestions.append("Използване на по-реалистични грешки")
        
        if any("CMB отклонение" in p for p in problems):
            suggestions.append("Подобрение на CMB angular scale изчисление")
            suggestions.append("Проверка на recombination redshift")
            suggestions.append("Корекция на sound horizon за open universe")
        
        if any("Систематично отклонение" in p for p in problems):
            suggestions.append("Добавяне на bias параметър")
            suggestions.append("Калибриране на теоретичните предсказания")
            suggestions.append("Проверка на космологичните параметри")
        
        if any("Redshift зависимост" in p for p in problems):
            suggestions.append("Подобрение на redshift еволюция")
            suggestions.append("Проверка на нелинейни ефекти")
            suggestions.append("Корекция на анизотропни ефекти")
        
        if any("Огромно индивидуално" in p for p in problems):
            suggestions.append("Идентифициране на problematic data points")
            suggestions.append("Проверка на individual survey systematics")
            suggestions.append("Възможно премахване на outliers")
        
        # Общи предложения
        suggestions.append("Използване на MCMC за по-добро параметър estimation")
        suggestions.append("Bootstrap анализ за uncertainty estimation")
        suggestions.append("Cross-validation с независими данни")
        
        print(f"💡 ПРЕДЛОЖЕНИЯ ({len(suggestions)}):")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
        
        return suggestions
    
    def create_residuals_plots(self, bao_results: Dict, cmb_results: Dict):
        """Създаване на plots за остатъци"""
        print("\n📈 СЪЗДАВАНЕ НА RESIDUALS PLOTS")
        print("=" * 40)
        
        # Настройка на plot стил
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Residuals Analysis: No-Λ Cosmology', fontsize=16, fontweight='bold')
        
        # 1. BAO остатъци vs redshift
        ax1 = axes[0, 0]
        ax1.errorbar(bao_results['redshifts'], bao_results['residuals'], 
                    yerr=bao_results['errors'], fmt='o', capsize=5, alpha=0.7)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Redshift z')
        ax1.set_ylabel('Residuals (obs - theory)')
        ax1.set_title('BAO Residuals vs Redshift')
        ax1.grid(True, alpha=0.3)
        
        # 2. Normalized остатъци
        ax2 = axes[0, 1]
        ax2.scatter(bao_results['redshifts'], bao_results['normalized_residuals'], 
                   alpha=0.7, s=50)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=2, color='orange', linestyle=':', alpha=0.5, label='2σ')
        ax2.axhline(y=-2, color='orange', linestyle=':', alpha=0.5)
        ax2.axhline(y=3, color='red', linestyle=':', alpha=0.5, label='3σ')
        ax2.axhline(y=-3, color='red', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Redshift z')
        ax2.set_ylabel('Normalized Residuals (σ)')
        ax2.set_title('Normalized BAO Residuals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot за нормалност
        ax3 = axes[1, 0]
        stats.probplot(bao_results['normalized_residuals'], dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot: Normality Test')
        ax3.grid(True, alpha=0.3)
        
        # 4. Histogram на остатъци
        ax4 = axes[1, 1]
        ax4.hist(bao_results['normalized_residuals'], bins=8, alpha=0.7, 
                density=True, edgecolor='black')
        
        # Overlay нормално разпределение
        x = np.linspace(-4, 4, 100)
        ax4.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, 
                label='Standard Normal')
        ax4.set_xlabel('Normalized Residuals')
        ax4.set_ylabel('Density')
        ax4.set_title('Residuals Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('residuals_analysis_plots.png', dpi=300, bbox_inches='tight')
        print("✅ Plots записани: residuals_analysis_plots.png")
        
        # CMB residuals info
        print(f"\n📊 CMB RESIDUALS INFO:")
        print(f"CMB χ²: {cmb_results['chi2_cmb']:.1f}")
        print(f"CMB σ отклонение: {cmb_results['normalized_residual']:.1f}")
        
        return fig
    
    def save_results(self, bao_results: Dict, cmb_results: Dict, 
                    problems: List[str], suggestions: List[str]):
        """Записване на резултатите"""
        print("\n💾 ЗАПИСВАНЕ НА РЕЗУЛТАТИТЕ")
        print("=" * 40)
        
        # Detailed results
        results_text = f"""
RESIDUALS ANALYSIS REPORT
========================

🎯 OBJECTIVE: Анализ на остатъци за χ² = 223.749

📊 SUMMARY STATISTICS:
- BAO χ²: {bao_results['chi2_total']:.1f}
- CMB χ²: {cmb_results['chi2_cmb']:.1f}  
- Total χ²: {bao_results['chi2_total'] + cmb_results['chi2_cmb']:.1f}
- Reduced χ²: {(bao_results['chi2_total'] + cmb_results['chi2_cmb']) / 10:.1f}

📈 BAO ANALYSIS:
- Брой точки: {len(bao_results['residuals'])}
- Средни остатъци: {np.mean(bao_results['residuals']):.3f}
- Стандартно отклонение: {np.std(bao_results['residuals']):.3f}
- Максимален остатък: {np.max(np.abs(bao_results['residuals'])):.3f}

📈 CMB ANALYSIS:
- Наблюдавано θₛ: {cmb_results['theta_s_obs']:.6f}
- Теоретично θₛ: {cmb_results['theta_s_theory']:.6f}
- Остатък: {cmb_results['residual']:.6f}
- Нормализиран остатък: {cmb_results['normalized_residual']:.1f}σ

🚨 IDENTIFIED PROBLEMS ({len(problems)}):
"""
        
        for i, problem in enumerate(problems, 1):
            results_text += f"{i}. {problem}\n"
        
        results_text += f"\n💡 SUGGESTED IMPROVEMENTS ({len(suggestions)}):\n"
        for i, suggestion in enumerate(suggestions, 1):
            results_text += f"{i}. {suggestion}\n"
        
        results_text += f"""
🔍 DETAILED BAO RESIDUALS:
z     | Observed | Theory  | Residual | χ²
------|----------|---------|----------|--------
"""
        
        for i, z in enumerate(bao_results['redshifts']):
            results_text += f"{z:.3f} | {bao_results['observed'][i]:8.3f} | {bao_results['predicted'][i]:7.3f} | {bao_results['residuals'][i]:8.3f} | {bao_results['chi2_individual'][i]:6.1f}\n"
        
        # Записване на файл
        with open('residuals_analysis_results.txt', 'w', encoding='utf-8') as f:
            f.write(results_text)
        
        # CSV файл с данни
        df = pd.DataFrame({
            'redshift': bao_results['redshifts'],
            'observed': bao_results['observed'],
            'predicted': bao_results['predicted'],
            'error': bao_results['errors'],
            'residual': bao_results['residuals'],
            'normalized_residual': bao_results['normalized_residuals'],
            'chi2_individual': bao_results['chi2_individual']
        })
        
        df.to_csv('bao_residuals_data.csv', index=False)
        
        print("✅ Резултати записани:")
        print("   📋 residuals_analysis_results.txt")
        print("   📊 bao_residuals_data.csv")
        print("   📈 residuals_analysis_plots.png")

def main():
    """Главна функция"""
    print("🔍 RESIDUALS ANALYSIS")
    print("🎯 Цел: Анализ на остатъци за χ² = 223.749")
    print("=" * 50)
    
    # Създаване на анализатор
    analyzer = ResidualsAnalyzer()
    
    # Анализ на BAO остатъци
    bao_results = analyzer.analyze_bao_residuals()
    
    # Анализ на CMB остатъци
    cmb_results = analyzer.analyze_cmb_residuals()
    
    # Статистически тестове
    stat_tests = analyzer.statistical_tests(bao_results, cmb_results)
    
    # Идентифициране на проблеми
    problems = analyzer.identify_problems(bao_results, cmb_results)
    
    # Предложения за подобрения
    suggestions = analyzer.suggest_improvements(problems)
    
    # Създаване на plots
    analyzer.create_residuals_plots(bao_results, cmb_results)
    
    # Записване на резултатите
    analyzer.save_results(bao_results, cmb_results, problems, suggestions)
    
    print("\n🎉 Residuals анализът завърши!")
    print("📋 Проверете записаните файлове за детайли")

if __name__ == "__main__":
    main() 