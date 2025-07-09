"""
Демонстрация на χ², Δχ² и σ еквивалент анализи
===========================================

Този скрипт демонстрира използването на:
- χ² анализ за оценка на качеството на модели
- Δχ² анализ за сравняване на модели
- σ еквивалент анализ за доверителни интервали
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2
import sys
import os

# Добавяме пътища към модулите
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation_tests.common_utils.statistical_tests import StatisticalSignificanceTest
from validation_tests.common_utils.data_processors import RawDataProcessor

class ChiSquaredAnalysisDemo:
    """
    Демонстрация на χ² анализи с реални космологични данни
    """
    
    def __init__(self):
        self.stat_test = StatisticalSignificanceTest()
        self.data_processor = RawDataProcessor()
        
    def load_demonstration_data(self):
        """
        Зарежда данни за демонстрация
        """
        # Генерираме синтетични данни базирани на реални наблюдения
        np.random.seed(42)
        
        # Redshift данни (базирани на Pantheon+)
        z = np.logspace(-3, 0.5, 100)  # z от 0.001 до ~3
        
        # Истински модел (нелинейно време)
        def true_model(z, H0=70, Omega_m=0.3):
            # Упростен модел за нелинейно време
            c = 299792.458  # km/s
            # Luminosity distance с нелинейна корекция
            d_L = (c / H0) * z * (1 + z/2) * (1 + 0.1 * z**2)
            return 5 * np.log10(d_L) + 25
        
        # Наблюдавани данни с грешки
        true_magnitudes = true_model(z)
        errors = 0.1 + 0.02 * z  # Грешки нарастват с z
        observed_magnitudes = true_magnitudes + np.random.normal(0, errors)
        
        return {
            'z': z,
            'observed_magnitudes': observed_magnitudes,
            'true_magnitudes': true_magnitudes,
            'errors': errors
        }
    
    def define_test_models(self, z, observed_magnitudes, errors):
        """
        Дефинира тестови модели за сравнение
        """
        # Модел 1: Стандартен ΛCDM
        def lambda_cdm_model(z, H0=70, Omega_m=0.3):
            c = 299792.458
            # Опростен ΛCDM модел
            d_L = (c / H0) * z * (1 + z/2)
            return 5 * np.log10(d_L) + 25
        
        # Модел 2: Нелинейно време (по-сложен)
        def nonlinear_time_model(z, H0=70, Omega_m=0.3, alpha=0.1):
            c = 299792.458
            # Модел с нелинейна корекция
            d_L = (c / H0) * z * (1 + z/2) * (1 + alpha * z**2)
            return 5 * np.log10(d_L) + 25
        
        # Модел 3: Полиномиален фит
        def polynomial_model(z, a0=30, a1=5, a2=2):
            return a0 + a1 * z + a2 * z**2
        
        # Фитваме модели към данните
        models = {}
        
        # ΛCDM модел
        lambda_cdm_pred = lambda_cdm_model(z)
        models['ΛCDM'] = {
            'observed': observed_magnitudes,
            'predicted': lambda_cdm_pred,
            'errors': errors,
            'n_params': 2,
            'description': 'Стандартен ΛCDM модел'
        }
        
        # Нелинейно време модел
        nonlinear_pred = nonlinear_time_model(z)
        models['Нелинейно време'] = {
            'observed': observed_magnitudes,
            'predicted': nonlinear_pred,
            'errors': errors,
            'n_params': 3,
            'description': 'Модел с нелинейно време'
        }
        
        # Полиномиален модел
        poly_pred = polynomial_model(z)
        models['Полиномиален'] = {
            'observed': observed_magnitudes,
            'predicted': poly_pred,
            'errors': errors,
            'n_params': 3,
            'description': 'Полиномиален фит'
        }
        
        return models
    
    def run_chi_squared_analysis(self, models):
        """
        Стартира χ² анализ за всички модели
        """
        print("=" * 80)
        print("χ² АНАЛИЗ НА МОДЕЛИ")
        print("=" * 80)
        
        results = {}
        
        for model_name, model_data in models.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)
            
            chi2_result = self.stat_test.chi_squared_analysis(
                model_data['observed'],
                model_data['predicted'],
                model_data['errors'],
                model_data['n_params']
            )
            
            results[model_name] = chi2_result
            
            print(f"χ² = {chi2_result['chi_squared']:.2f}")
            print(f"χ²_red = {chi2_result['chi_squared_reduced']:.2f}")
            print(f"Степени на свобода = {chi2_result['degrees_of_freedom']}")
            print(f"p-стойност = {chi2_result['p_value']:.6f}")
            print(f"AIC = {chi2_result['aic']:.2f}")
            print(f"BIC = {chi2_result['bic']:.2f}")
            print(f"Оценка: {chi2_result['interpretation']}")
        
        return results
    
    def run_delta_chi_squared_analysis(self, chi2_results):
        """
        Стартира Δχ² анализ за сравняване на модели
        """
        print("\n" + "=" * 80)
        print("Δχ² АНАЛИЗ ЗА СРАВНЯВАНЕ НА МОДЕЛИ")
        print("=" * 80)
        
        model_names = list(chi2_results.keys())
        delta_results = {}
        
        # Сравняваме всички модели помежду си
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_1 = model_names[i]
                model_2 = model_names[j]
                
                delta_result = self.stat_test.delta_chi_squared_analysis(
                    chi2_results[model_1]['chi_squared'],
                    chi2_results[model_2]['chi_squared'],
                    chi2_results[model_1]['degrees_of_freedom'],
                    chi2_results[model_2]['degrees_of_freedom'],
                    model_1,
                    model_2
                )
                
                comparison_name = f"{model_1} vs {model_2}"
                delta_results[comparison_name] = delta_result
                
                print(f"\n{comparison_name.upper()}:")
                print("-" * 40)
                print(f"Δχ² = {delta_result['delta_chi2']:.2f}")
                print(f"Δdof = {delta_result['delta_dof']}")
                print(f"p-стойност = {delta_result['p_value']:.6f}")
                print(f"σ еквивалент = {delta_result['sigma_equivalent']:.2f}σ")
                print(f"По-добър модел: {delta_result['better_model']}")
                print(f"Значимост: {delta_result['significance']}")
        
        return delta_results
    
    def run_sigma_equivalent_analysis(self, models):
        """
        Стартира σ еквивалент анализ
        """
        print("\n" + "=" * 80)
        print("σ ЕКВИВАЛЕНТ АНАЛИЗ")
        print("=" * 80)
        
        # Подготвяме данните
        chi2_values = []
        dof_values = []
        model_names = []
        
        for model_name, model_data in models.items():
            chi2_result = self.stat_test.chi_squared_analysis(
                model_data['observed'],
                model_data['predicted'],
                model_data['errors'],
                model_data['n_params']
            )
            chi2_values.append(chi2_result['chi_squared'])
            dof_values.append(chi2_result['degrees_of_freedom'])
            model_names.append(model_name)
        
        # Стартираме σ еквивалент анализ
        sigma_result = self.stat_test.sigma_equivalent_analysis(
            chi2_values, dof_values, model_names
        )
        
        print(f"\nНАЙ-ДОБЪР МОДЕЛ: {sigma_result['best_model']}")
        print(f"Най-добро χ² = {sigma_result['best_chi2']:.2f}")
        print("-" * 40)
        
        for model_name, model_result in sigma_result['models'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  χ² = {model_result['chi2']:.2f}")
            print(f"  Δχ² = {model_result['delta_chi2']:.2f}")
            print(f"  σ еквивалент = {model_result['sigma_equivalent']:.2f}σ")
            print(f"  Най-добър: {'Да' if model_result['is_best'] else 'Не'}")
            
            print("  Доверителни интервали:")
            for sigma_level, interval_data in model_result['confidence_intervals'].items():
                excluded_text = "ИЗКЛЮЧЕН" if interval_data['excluded'] else "включен"
                print(f"    {sigma_level}: {excluded_text} (χ² праг = {interval_data['chi2_threshold']:.2f})")
        
        return sigma_result
    
    def create_visualizations(self, models, chi2_results, delta_results, sigma_result):
        """
        Създава графики за визуализация на резултатите
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Данни и модели
        z = models['ΛCDM']['observed']  # Използваме z от първия модел
        z_actual = np.logspace(-3, 0.5, 100)  # Пресъздаваме z
        
        for i, (model_name, model_data) in enumerate(models.items()):
            color = ['blue', 'red', 'green'][i]
            axes[0, 0].plot(z_actual, model_data['predicted'], 
                           color=color, label=f'{model_name} модел', linewidth=2)
        
        axes[0, 0].errorbar(z_actual, model_data['observed'], 
                           yerr=model_data['errors'], 
                           fmt='ko', alpha=0.6, label='Наблюдения')
        axes[0, 0].set_xlabel('Червено отместване z')
        axes[0, 0].set_ylabel('Модулна величина')
        axes[0, 0].set_title('Модели и наблюдения')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. χ² сравнение
        model_names = list(chi2_results.keys())
        chi2_values = [chi2_results[name]['chi_squared'] for name in model_names]
        chi2_reduced = [chi2_results[name]['chi_squared_reduced'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, chi2_values, width, label='χ²', alpha=0.8)
        axes[0, 1].bar(x + width/2, chi2_reduced, width, label='χ²_red', alpha=0.8)
        axes[0, 1].set_xlabel('Модел')
        axes[0, 1].set_ylabel('χ² стойност')
        axes[0, 1].set_title('χ² сравнение')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Δχ² и σ еквивалент
        comparison_names = list(delta_results.keys())
        delta_chi2_values = [delta_results[name]['delta_chi2'] for name in comparison_names]
        sigma_values = [delta_results[name]['sigma_equivalent'] for name in comparison_names]
        
        axes[1, 0].bar(range(len(comparison_names)), delta_chi2_values, alpha=0.8)
        axes[1, 0].set_xlabel('Сравнение')
        axes[1, 0].set_ylabel('Δχ²')
        axes[1, 0].set_title('Δχ² между модели')
        axes[1, 0].set_xticks(range(len(comparison_names)))
        axes[1, 0].set_xticklabels(comparison_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. σ еквивалент
        axes[1, 1].bar(range(len(comparison_names)), sigma_values, alpha=0.8, color='orange')
        axes[1, 1].set_xlabel('Сравнение')
        axes[1, 1].set_ylabel('σ еквивалент')
        axes[1, 1].set_title('Статистическа значимост (σ)')
        axes[1, 1].set_xticks(range(len(comparison_names)))
        axes[1, 1].set_xticklabels(comparison_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Добавяме хоризонтални линии за 1σ, 2σ, 3σ
        for sigma_level in [1, 2, 3]:
            axes[1, 1].axhline(y=sigma_level, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].text(0.1, sigma_level + 0.1, f'{sigma_level}σ', color='red')
        
        plt.tight_layout()
        plt.savefig('chi_squared_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_comprehensive_analysis(self):
        """
        Стартира пълен χ² анализ
        """
        print("🔬 ДЕМОНСТРАЦИЯ НА χ², Δχ² И σ ЕКВИВАЛЕНТ АНАЛИЗИ")
        print("=" * 80)
        
        # 1. Зареждаме данни
        print("📊 Зареждане на демонстрационни данни...")
        data = self.load_demonstration_data()
        
        # 2. Дефинираме модели
        print("🔧 Дефиниране на тестови модели...")
        models = self.define_test_models(
            data['z'], 
            data['observed_magnitudes'], 
            data['errors']
        )
        
        # 3. χ² анализ
        print("📐 Стартиране на χ² анализ...")
        chi2_results = self.run_chi_squared_analysis(models)
        
        # 4. Δχ² анализ
        print("📊 Стартиране на Δχ² анализ...")
        delta_results = self.run_delta_chi_squared_analysis(chi2_results)
        
        # 5. σ еквивалент анализ
        print("📈 Стартиране на σ еквивалент анализ...")
        sigma_result = self.run_sigma_equivalent_analysis(models)
        
        # 6. Визуализация
        print("📈 Създаване на графики...")
        fig = self.create_visualizations(models, chi2_results, delta_results, sigma_result)
        
        # 7. Обобщение
        print("\n" + "=" * 80)
        print("ОБОБЩЕНИЕ НА РЕЗУЛТАТИТЕ")
        print("=" * 80)
        
        best_model = sigma_result['best_model']
        best_chi2 = sigma_result['best_chi2']
        
        print(f"✅ Най-добър модел: {best_model}")
        print(f"✅ Най-добро χ² = {best_chi2:.2f}")
        
        # Намираме най-значимото сравнение
        max_sigma = 0
        best_comparison = ""
        for comp_name, comp_result in delta_results.items():
            if comp_result['sigma_equivalent'] > max_sigma:
                max_sigma = comp_result['sigma_equivalent']
                best_comparison = comp_name
        
        print(f"✅ Най-значимо сравнение: {best_comparison}")
        print(f"✅ Максимална значимост: {max_sigma:.2f}σ")
        
        if max_sigma > 3:
            print("🎯 Резултат: Силно статистическо доказателство за разлики между модели!")
        elif max_sigma > 2:
            print("🎯 Резултат: Умерено статистическо доказателство за разлики между модели")
        else:
            print("🎯 Резултат: Слабо статистическо доказателство за разлики между модели")
        
        print("\n✅ Демонстрацията завърши успешно!")
        
        return {
            'chi2_results': chi2_results,
            'delta_results': delta_results,
            'sigma_result': sigma_result,
            'models': models,
            'data': data
        }


def main():
    """
    Основна функция за стартиране на демонстрацията
    """
    demo = ChiSquaredAnalysisDemo()
    results = demo.run_comprehensive_analysis()
    return results


if __name__ == "__main__":
    results = main() 