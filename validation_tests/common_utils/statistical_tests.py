"""
Статистически тестове и тестване на значимост
===========================================

Този модул имплементира:
- Статистическа значимост на модели
- Тестове за остатъчен шум
- Goodness-of-fit тестове
- Cross-validation
- Bootstrap анализ

Автор: Система за анализ на нелинейно време
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kstest, shapiro, anderson, chi2, f
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from typing import Callable, Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore')


class StatisticalSignificanceTest:
    """
    Клас за изчисляване на статистическа значимост на резултатите
    """
    
    def __init__(self):
        self.results = {}
        
    def chi_squared_analysis(self, observed: np.ndarray, predicted: np.ndarray, 
                           errors: Optional[np.ndarray] = None, 
                           n_params: int = 1) -> Dict[str, Any]:
        """
        Пълен χ² анализ на модела
        
        Args:
            observed: Наблюдавани стойности
            predicted: Предсказани стойности
            errors: Грешки в наблюденията (ако няма - използва се остатъчна дисперсия)
            n_params: Брой параметри в модела
            
        Returns:
            Резултати от χ² анализа
        """
        # Пресмятаме остатъци
        residuals = observed - predicted
        n_data = len(observed)
        
        # Ако няма грешки, използваме остатъчната дисперсия
        if errors is None:
            errors = np.std(residuals) * np.ones_like(residuals)
        
        # Избягваме деление на нула
        errors = np.where(errors == 0, np.std(residuals), errors)
        
        # Пресмятаме χ²
        chi_squared = np.sum((residuals / errors) ** 2)
        
        # Степени на свобода
        dof = n_data - n_params
        
        # Редуцирано χ²
        chi_squared_reduced = chi_squared / dof if dof > 0 else np.inf
        
        # p-стойност
        p_value = 1 - chi2.cdf(chi_squared, dof) if dof > 0 else 0
        
        # Akaike Information Criterion (AIC)
        aic = chi_squared + 2 * n_params
        
        # Bayesian Information Criterion (BIC)
        bic = chi_squared + n_params * np.log(n_data)
        
        # Интерпретация
        if chi_squared_reduced < 1:
            interpretation = "Отличен модел (χ²_red < 1)"
        elif chi_squared_reduced < 2:
            interpretation = "Добър модел (χ²_red < 2)"
        elif chi_squared_reduced < 5:
            interpretation = "Приемлив модел (χ²_red < 5)"
        else:
            interpretation = "Лош модел (χ²_red > 5)"
        
        return {
            'chi_squared': chi_squared,
            'chi_squared_reduced': chi_squared_reduced,
            'degrees_of_freedom': dof,
            'p_value': p_value,
            'aic': aic,
            'bic': bic,
            'n_data': n_data,
            'n_params': n_params,
            'interpretation': interpretation,
            'residuals': residuals,
            'errors': errors
        }
    
    def delta_chi_squared_analysis(self, chi2_1: float, chi2_2: float, 
                                 dof_1: int, dof_2: int, 
                                 model_1_name: str = "Модел 1",
                                 model_2_name: str = "Модел 2") -> Dict[str, Any]:
        """
        Δχ² анализ за сравняване на модели
        
        Args:
            chi2_1: χ² на първия модел
            chi2_2: χ² на втория модел
            dof_1: Степени на свобода на първия модел
            dof_2: Степени на свобода на втория модел
            model_1_name: Име на първия модел
            model_2_name: Име на втория модел
            
        Returns:
            Резултати от Δχ² анализа
        """
        # Δχ² и Δdof
        delta_chi2 = abs(chi2_1 - chi2_2)
        delta_dof = abs(dof_1 - dof_2)
        
        # Кой модел е по-добър
        if chi2_1 < chi2_2:
            better_model = model_1_name
            worse_model = model_2_name
        else:
            better_model = model_2_name
            worse_model = model_1_name
        
        # p-стойност за разликата
        p_value = 1 - chi2.cdf(delta_chi2, delta_dof) if delta_dof > 0 else 0
        
        # Статистическа значимост
        sigma_equivalent = self._chi2_to_sigma(delta_chi2, delta_dof)
        
        # Интерпретация
        if sigma_equivalent < 1:
            significance = "Незначителна разлика (< 1σ)"
        elif sigma_equivalent < 2:
            significance = "Слаба разлика (1-2σ)"
        elif sigma_equivalent < 3:
            significance = "Умерена разлика (2-3σ)"
        elif sigma_equivalent < 4:
            significance = "Силна разлика (3-4σ)"
        else:
            significance = "Много силна разлика (> 4σ)"
        
        return {
            'delta_chi2': delta_chi2,
            'delta_dof': delta_dof,
            'p_value': p_value,
            'sigma_equivalent': sigma_equivalent,
            'better_model': better_model,
            'worse_model': worse_model,
            'significance': significance,
            'chi2_1': chi2_1,
            'chi2_2': chi2_2,
            'dof_1': dof_1,
            'dof_2': dof_2
        }
    
    def sigma_equivalent_analysis(self, chi2_values: List[float], 
                                dof_values: List[int],
                                model_names: List[str],
                                confidence_levels: List[float] = [0.68, 0.95, 0.997]) -> Dict[str, Any]:
        """
        Пълен σ еквивалент анализ
        
        Args:
            chi2_values: Списък с χ² стойности
            dof_values: Списък със степени на свобода
            model_names: Имена на моделите
            confidence_levels: Доверителни интервали (0.68 = 1σ, 0.95 = 2σ, 0.997 = 3σ)
            
        Returns:
            Резултати от σ еквивалент анализа
        """
        results = {}
        
        # Намираме най-добрия модел (минимално χ²)
        best_index = np.argmin(chi2_values)
        best_chi2 = chi2_values[best_index]
        best_dof = dof_values[best_index]
        best_model = model_names[best_index]
        
        # Анализираме всички модели
        for i, (chi2_val, dof_val, model_name) in enumerate(zip(chi2_values, dof_values, model_names)):
            
            # Δχ² спрямо най-добрия модел
            delta_chi2 = chi2_val - best_chi2
            delta_dof = dof_val - best_dof
            
            # σ еквивалент
            sigma_eq = self._chi2_to_sigma(delta_chi2, max(1, abs(delta_dof)))
            
            # Доверителни интервали
            confidence_intervals = {}
            for cl in confidence_levels:
                sigma_level = self._confidence_to_sigma(cl)
                chi2_threshold = best_chi2 + chi2.ppf(cl, max(1, abs(delta_dof)))
                confidence_intervals[f"{sigma_level:.0f}σ"] = {
                    'confidence_level': cl,
                    'chi2_threshold': chi2_threshold,
                    'excluded': chi2_val > chi2_threshold
                }
            
            results[model_name] = {
                'chi2': chi2_val,
                'dof': dof_val,
                'delta_chi2': delta_chi2,
                'sigma_equivalent': sigma_eq,
                'confidence_intervals': confidence_intervals,
                'is_best': i == best_index
            }
        
        return {
            'best_model': best_model,
            'best_chi2': best_chi2,
            'models': results,
            'confidence_levels': confidence_levels
        }
    
    def _chi2_to_sigma(self, chi2_val: float, dof: int) -> float:
        """
        Преобразува χ² в σ еквивалент
        
        Args:
            chi2_val: χ² стойност
            dof: Степени на свобода
            
        Returns:
            σ еквивалент
        """
        if dof <= 0:
            return 0
        
        # За 1 степен на свобода - директна формула
        if dof == 1:
            return np.sqrt(chi2_val)
        
        # За повече степени на свобода - използваме p-стойност
        p_value = 1 - chi2.cdf(chi2_val, dof)
        
        # Защитаваме от численни грешки
        if p_value <= 0:
            return 5  # Много висока значимост
        if p_value >= 1:
            return 0
        
        # Преобразуваме p-стойност в σ еквивалент
        return abs(stats.norm.ppf(p_value / 2))
    
    def _confidence_to_sigma(self, confidence_level: float) -> float:
        """
        Преобразува доверителен интервал в σ еквивалент
        
        Args:
            confidence_level: Доверителен интервал (0.68, 0.95, 0.997)
            
        Returns:
            σ еквивалент
        """
        alpha = 1 - confidence_level
        return abs(stats.norm.ppf(alpha / 2))
    
    def model_comparison_full_analysis(self, models_data: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Пълен модел сравнение с всички статистически тестове
        
        Args:
            models_data: Речник с данни за моделите
                        {model_name: {'observed': array, 'predicted': array, 'errors': array, 'n_params': int}}
        
        Returns:
            Пълни резултати от анализа
        """
        results = {}
        chi2_values = []
        dof_values = []
        model_names = []
        
        # Анализираме всеки модел
        for model_name, data in models_data.items():
            chi2_analysis = self.chi_squared_analysis(
                data['observed'], 
                data['predicted'], 
                data.get('errors'),
                data['n_params']
            )
            
            results[model_name] = chi2_analysis
            chi2_values.append(chi2_analysis['chi_squared'])
            dof_values.append(chi2_analysis['degrees_of_freedom'])
            model_names.append(model_name)
        
        # Δχ² анализ (сравняваме всички с най-добрия)
        best_index = np.argmin(chi2_values)
        delta_chi2_results = {}
        
        for i, model_name in enumerate(model_names):
            if i != best_index:
                delta_analysis = self.delta_chi_squared_analysis(
                    chi2_values[best_index], chi2_values[i],
                    dof_values[best_index], dof_values[i],
                    model_names[best_index], model_name
                )
                delta_chi2_results[f"{model_names[best_index]}_vs_{model_name}"] = delta_analysis
        
        # σ еквивалент анализ
        sigma_analysis = self.sigma_equivalent_analysis(chi2_values, dof_values, model_names)
        
        return {
            'individual_models': results,
            'delta_chi2_comparisons': delta_chi2_results,
            'sigma_equivalent_analysis': sigma_analysis,
            'summary': {
                'best_model': model_names[best_index],
                'best_chi2': chi2_values[best_index],
                'best_chi2_reduced': results[model_names[best_index]]['chi_squared_reduced'],
                'n_models': len(model_names),
                'model_names': model_names
            }
        }
    
    def kolmogorov_smirnov_test(self,
                              residuals: np.ndarray,
                              distribution: str = 'norm') -> Dict[str, Any]:
        """
        Kolmogorov-Smirnov тест за нормалност на остатъците
        
        Args:
            residuals: Остатъци от модел
            distribution: Теоретично разпределение ('norm', 'uniform', etc.)
            
        Returns:
            Резултати от теста
        """
        # Нормализираме остатъците
        residuals_normalized = (residuals - np.mean(residuals)) / np.std(residuals)
        
        # KS тест
        if distribution == 'norm':
            ks_stat, p_value = kstest(residuals_normalized, 'norm')
        elif distribution == 'uniform':
            ks_stat, p_value = kstest(residuals_normalized, 'uniform')
        else:
            raise ValueError(f"Неподдържано разпределение: {distribution}")
        
        # Заключение
        reject_null = p_value < 0.05 # По подразбиране ниво на значимост
        
        result = {
            'test_name': f'Kolmogorov-Smirnov ({distribution})',
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'reject_null': reject_null,
            'significance_level': 0.05,
            'interpretation': f'Остатъците не следват {distribution} разпределение' if reject_null else f'Остатъците следват {distribution} разпределение'
        }
        
        self.results['ks_test'] = result
        return result
    
    def shapiro_wilk_test(self, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Shapiro-Wilk тест за нормалност
        
        Args:
            residuals: Остатъци от модел
            
        Returns:
            Резултати от теста
        """
        # Shapiro-Wilk тест
        sw_stat, p_value = shapiro(residuals)
        
        # Заключение
        reject_null = p_value < 0.05 # По подразбиране ниво на значимост
        
        result = {
            'test_name': 'Shapiro-Wilk',
            'sw_statistic': sw_stat,
            'p_value': p_value,
            'reject_null': reject_null,
            'significance_level': 0.05,
            'interpretation': 'Остатъците не следват нормално разпределение' if reject_null else 'Остатъците следват нормално разпределение'
        }
        
        self.results['shapiro_wilk'] = result
        return result
    
    def anderson_darling_test(self, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Anderson-Darling тест за нормалност
        
        Args:
            residuals: Остатъци от модел
            
        Returns:
            Резултати от теста
        """
        # Anderson-Darling тест
        ad_result = anderson(residuals, dist='norm')
        
        # Намираме съответната критична стойност
        critical_values = ad_result.critical_values
        
        # Проверяваме дали има significance_levels атрибут
        if hasattr(ad_result, 'significance_levels'):
            significance_levels = ad_result.significance_levels
            # Намираме най-близкото ниво на значимост
            closest_idx = np.argmin(np.abs(significance_levels - 0.05 * 100)) # По подразбиране ниво на значимост
            critical_value = critical_values[closest_idx]
        else:
            # За по-новите версии на scipy, използваме стандартни нива
            significance_levels = [15, 10, 5, 2.5, 1]  # Процентни нива
            if 0.05 * 100 <= 1:
                critical_value = critical_values[4]  # 1%
            elif 0.05 * 100 <= 2.5:
                critical_value = critical_values[3]  # 2.5%
            elif 0.05 * 100 <= 5:
                critical_value = critical_values[2]  # 5%
            elif 0.05 * 100 <= 10:
                critical_value = critical_values[1]  # 10%
            else:
                critical_value = critical_values[0]  # 15%
        
        # Заключение
        reject_null = ad_result.statistic > critical_value
        
        result = {
            'test_name': 'Anderson-Darling',
            'ad_statistic': ad_result.statistic,
            'critical_value': critical_value,
            'reject_null': reject_null,
            'significance_level': 0.05,
            'interpretation': 'Остатъците не следват нормално разпределение' if reject_null else 'Остатъците следват нормално разпределение'
        }
        
        self.results['anderson_darling'] = result
        return result
    
    def f_test_model_comparison(self,
                              rss1: float,
                              rss2: float,
                              df1: int,
                              df2: int,
                              n: int) -> Dict[str, Any]:
        """
        F-тест за сравнение на модели
        
        Args:
            rss1: Остатъчна сума на квадратите за модел 1
            rss2: Остатъчна сума на квадратите за модел 2
            df1: Степени на свобода за модел 1
            df2: Степени на свобода за модел 2
            n: Брой наблюдения
            
        Returns:
            Резултати от теста
        """
        # F-статистика
        f_stat = ((rss1 - rss2) / (df2 - df1)) / (rss2 / (n - df2 - 1))
        
        # p-стойност
        p_value = 1 - stats.f.cdf(f_stat, df2 - df1, n - df2 - 1)
        
        # Критична стойност
        critical_value = stats.f.ppf(1 - 0.05, df2 - df1, n - df2 - 1) # По подразбиране ниво на значимост
        
        # Заключение
        reject_null = f_stat > critical_value
        
        result = {
            'test_name': 'F-test Model Comparison',
            'f_statistic': f_stat,
            'p_value': p_value,
            'critical_value': critical_value,
            'reject_null': reject_null,
            'significance_level': 0.05,
            'interpretation': 'По-сложният модел е значимо по-добър' if reject_null else 'Разликата между модели не е значима'
        }
        
        self.results['f_test'] = result
        return result
    
    def runs_test(self, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Runs тест за автокорелация в остатъците
        
        Args:
            residuals: Остатъци от модел
            
        Returns:
            Резултати от теста
        """
        # Превръщаме остатъците в + и - знаци
        signs = np.sign(residuals)
        
        # Броим runs (последователни еднакви знаци)
        runs = 1
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1]:
                runs += 1
        
        # Очакван брой runs
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)
        n_total = len(signs)
        
        expected_runs = (2 * n_pos * n_neg) / n_total + 1
        
        # Дисперсия на runs
        variance_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_total)) / (n_total**2 * (n_total - 1))
        
        # Z-статистика
        z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
        
        # p-стойност (двустранен тест)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Заключение
        reject_null = p_value < 0.05 # По подразбиране ниво на значимост
        
        result = {
            'test_name': 'Runs Test',
            'runs_observed': runs,
            'runs_expected': expected_runs,
            'z_statistic': z_stat,
            'p_value': p_value,
            'reject_null': reject_null,
            'significance_level': 0.05,
            'interpretation': 'Има автокорелация в остатъците' if reject_null else 'Няма автокорелация в остатъците'
        }
        
        self.results['runs_test'] = result
        return result
    
    def durbin_watson_test(self, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Durbin-Watson тест за автокорелация
        
        Args:
            residuals: Остатъци от модел
            
        Returns:
            Резултати от теста
        """
        # Durbin-Watson статистика
        diff = np.diff(residuals)
        dw_stat = np.sum(diff**2) / np.sum(residuals**2)
        
        # Интерпретация (приблизителна)
        if dw_stat < 1.5:
            interpretation = 'Положителна автокорелация'
        elif dw_stat > 2.5:
            interpretation = 'Отрицателна автокорелация'
        else:
            interpretation = 'Няма автокорелация'
        
        result = {
            'test_name': 'Durbin-Watson',
            'dw_statistic': dw_stat,
            'interpretation': interpretation
        }
        
        self.results['durbin_watson'] = result
        return result
    
    def comprehensive_residual_analysis(self,
                                      residuals: np.ndarray,
                                      fitted_values: np.ndarray = None) -> Dict[str, Any]:
        """
        Обширен анализ на остатъците
        
        Args:
            residuals: Остатъци от модел
            fitted_values: Прогнозни стойности
            
        Returns:
            Резултати от всички тестове
        """
        analysis = {}
        
        # Тестове за нормалност
        analysis['shapiro_wilk'] = self.shapiro_wilk_test(residuals)
        analysis['kolmogorov_smirnov'] = self.kolmogorov_smirnov_test(residuals)
        analysis['anderson_darling'] = self.anderson_darling_test(residuals)
        
        # Тестове за автокорелация
        analysis['runs_test'] = self.runs_test(residuals)
        analysis['durbin_watson'] = self.durbin_watson_test(residuals)
        
        # Основни статистики
        analysis['basic_stats'] = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'jarque_bera': stats.jarque_bera(residuals)
        }
        
        return analysis
    
    def plot_residual_diagnostics(self,
                                 residuals: np.ndarray,
                                 fitted_values: np.ndarray = None,
                                 save_path: str = None):
        """
        Графики за диагностика на остатъците
        
        Args:
            residuals: Остатъци от модел
            fitted_values: Прогнозни стойности
            save_path: Път за записване
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Хистограма на остатъците
        axes[0, 0].hist(residuals, bins=30, alpha=0.7, density=True, edgecolor='black')
        
        # Добавяме нормална крива
        x = np.linspace(residuals.min(), residuals.max(), 100)
        y = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
        axes[0, 0].plot(x, y, 'r-', linewidth=2, label='Нормално разпределение')
        
        axes[0, 0].set_xlabel('Остатъци')
        axes[0, 0].set_ylabel('Плътност')
        axes[0, 0].set_title('Хистограма на остатъците')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q plot
        stats.probplot(residuals, dist='norm', plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Нормалност)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Остатъци vs прогнозни стойности
        if fitted_values is not None:
            axes[1, 0].scatter(fitted_values, residuals, alpha=0.6)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Прогнозни стойности')
            axes[1, 0].set_ylabel('Остатъци')
            axes[1, 0].set_title('Остатъци vs Прогнозни стойности')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].plot(residuals, 'o-', alpha=0.6)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Наблюдение')
            axes[1, 0].set_ylabel('Остатъци')
            axes[1, 0].set_title('Остатъци vs Наблюдения')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Автокорелация
        from statsmodels.tsa.stattools import acf
        try:
            autocorr = acf(residuals, nlags=20)
            lags = np.arange(len(autocorr))
            axes[1, 1].bar(lags, autocorr, alpha=0.7)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Лаг')
            axes[1, 1].set_ylabel('Автокорелация')
            axes[1, 1].set_title('Автокорелационна функция')
            axes[1, 1].grid(True, alpha=0.3)
        except:
            # Ако statsmodels не е налично, правим прост график
            axes[1, 1].plot(residuals[:-1], residuals[1:], 'o', alpha=0.6)
            axes[1, 1].set_xlabel('Остатък(t)')
            axes[1, 1].set_ylabel('Остатък(t+1)')
            axes[1, 1].set_title('Лагова корелация')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Генерира текстов доклад с всички тестове
        
        Returns:
            Форматиран доклад
        """
        if not self.results:
            return "Няма резултати за показване"
        
        report = []
        report.append("=" * 60)
        report.append("ДОКЛАД ЗА СТАТИСТИЧЕСКА ЗНАЧИМОСТ")
        report.append("=" * 60)
        report.append("")
        
        for test_name, result in self.results.items():
            report.append(f"ТЕСТ: {result.get('test_name', test_name).upper()}")
            report.append("-" * 30)
            
            if 'p_value' in result:
                report.append(f"p-стойност: {result['p_value']:.6f}")
            
            if 'reject_null' in result:
                status = "ОТХВЪРЛЕНА" if result['reject_null'] else "НЕ ОТХВЪРЛЕНА"
                report.append(f"Нулева хипотеза: {status}")
            
            if 'interpretation' in result:
                report.append(f"Интерпретация: {result['interpretation']}")
            
            report.append("")
        
        return "\n".join(report)


class CrossValidationAnalysis:
    """
    Клас за cross-validation анализ
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Инициализация на cross-validation анализа
        
        Args:
            n_folds: Брой сгъвания
            random_state: Seed за възпроизводимост
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = {}
    
    def k_fold_validation(self,
                         model_func: Callable,
                         X: np.ndarray,
                         y: np.ndarray,
                         scoring: str = 'mse') -> Dict[str, Any]:
        """
        K-fold cross-validation
        
        Args:
            model_func: Функция на модела
            X: Независими променливи
            y: Зависима променлива
            scoring: Метрика за оценка
            
        Returns:
            Резултати от валидацията
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        scores = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Разделяме данните
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Тренираме модела
            model = model_func(X_train, y_train)
            
            # Прогнозираме
            y_pred = model(X_test)
            
            # Оценяваме
            if scoring == 'mse':
                score = mean_squared_error(y_test, y_pred)
            elif scoring == 'r2':
                score = r2_score(y_test, y_pred)
            elif scoring == 'mae':
                score = np.mean(np.abs(y_test - y_pred))
            else:
                raise ValueError(f"Неподдържана метрика: {scoring}")
            
            scores.append(score)
            fold_results.append({
                'fold': fold + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'score': score
            })
        
        result = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scoring_metric': scoring,
            'fold_results': fold_results
        }
        
        self.results['k_fold'] = result
        return result
    
    def plot_cv_results(self, save_path: str = None):
        """
        Графики на cross-validation резултати
        
        Args:
            save_path: Път за записване
        """
        if 'k_fold' not in self.results:
            print("Няма резултати за показване")
            return
        
        result = self.results['k_fold']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Box plot на резултатите
        ax1.boxplot(result['scores'])
        ax1.set_ylabel(f'{result["scoring_metric"].upper()} Score')
        ax1.set_title('Cross-Validation Score Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Резултати по fold
        folds = [r['fold'] for r in result['fold_results']]
        scores = [r['score'] for r in result['fold_results']]
        
        ax2.plot(folds, scores, 'o-', linewidth=2, markersize=8)
        ax2.axhline(y=result['mean_score'], color='r', linestyle='--', 
                   label=f'Средна стойност: {result["mean_score"]:.4f}')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel(f'{result["scoring_metric"].upper()} Score')
        ax2.set_title('Score по Fold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def test_statistical_significance():
    """
    Тестова функция за статистическа значимост
    """
    # Генерираме тестови данни
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y_true = 2 * x + 1
    noise = np.random.normal(0, 0.5, n)
    y_obs = y_true + noise
    
    # Простък linear fit
    coeffs = np.polyfit(x, y_obs, 1)
    y_pred = np.polyval(coeffs, x)
    residuals = y_obs - y_pred
    
    # Статистически тестове
    stat_test = StatisticalSignificanceTest()
    
    # Анализ на остатъците
    analysis = stat_test.comprehensive_residual_analysis(residuals, y_pred)
    
    # Генерираме доклад
    report = stat_test.generate_report()
    print(report)
    
    # Показваме графики
    stat_test.plot_residual_diagnostics(residuals, y_pred)


if __name__ == "__main__":
    test_statistical_significance() 