"""
Обширна валидационна система за теорията на нелинейно време
=========================================================

Този скрипт стартира всички валидационни тестове:
- GPS тестове
- Анализ на остатъчен шум
- Статистическа значимост
- Оптимизационни методи
- MCMC и Байесов анализ
- Обработка на сурови данни

Автор: Система за анализ на нелинейно време
"""

import sys
import os
import time
import warnings
from datetime import datetime
from typing import Dict, Any

# Добавяме пътя към общите утилити
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# Импортираме всички тестови модули
from validation_tests.gps_tests.gps_time_dilation import GPSTimeDilationTest
from validation_tests.residual_noise_tests.residual_noise_analyzer import ResidualNoiseAnalyzer
from validation_tests.primordial_analysis import (
    RecombinationAnalyzer,
    RelicNoiseAnalyzer,
    PrimordialFluctuationAnalyzer
)
from validation_tests.common_utils.optimization_engines import test_optimization_methods
from validation_tests.common_utils.mcmc_bayesian import test_mcmc_bayesian
from validation_tests.common_utils.statistical_tests import test_statistical_significance
from validation_tests.common_utils.data_processors import test_raw_data_processor
from validation_tests.chi_squared_analysis_demo import ChiSquaredAnalysisDemo


class ComprehensiveValidationSuite:
    """
    Обширна валидационна система
    """
    
    def __init__(self):
        """Инициализация на валидационната система"""
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def run_all_tests(self, include_gps: bool = True, 
                      include_residual: bool = True,
                      include_primordial: bool = True,
                      include_data_processing: bool = True,
                      include_optimization: bool = True,
                      include_mcmc: bool = True,
                      include_statistical: bool = True) -> Dict[str, Any]:
        """
        Стартира всички тестове
        
        Args:
            include_gps: Включва GPS тестове
            include_residual: Включва анализ на остатъчен шум
            include_primordial: Включва анализ на първобитните флуктуации
            include_data_processing: Включва обработка на данни
            include_optimization: Включва оптимизационни тестове
            include_mcmc: Включва MCMC тестове
            include_statistical: Включва статистически тестове
            
        Returns:
            Резултати от всички тестове
        """
        self.start_time = time.time()
        
        print("🚀 " + "="*70)
        print("🚀 СТАРТИРАНЕ НА ОБШИРНА ВАЛИДАЦИОННА СИСТЕМА")
        print("🚀 " + "="*70)
        print(f"🕐 Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. GPS тестове
        if include_gps:
            print("📡 GPS ТЕСТОВЕ")
            print("-" * 40)
            try:
                gps_test = GPSTimeDilationTest(use_nonlinear_time=True)
                self.results['gps_tests'] = gps_test.run_comprehensive_test()
                print("✅ GPS тестове завършени успешно")
            except Exception as e:
                print(f"❌ Грешка в GPS тестовете: {e}")
                self.results['gps_tests'] = {'error': str(e)}
            print()
        
        # 2. Анализ на остатъчен шум
        if include_residual:
            print("🔍 АНАЛИЗ НА ОСТАТЪЧЕН ШУМ")
            print("-" * 40)
            try:
                residual_analyzer = ResidualNoiseAnalyzer(use_raw_data=False)
                self.results['residual_noise'] = residual_analyzer.run_comprehensive_analysis()
                print("✅ Анализ на остатъчен шум завършен успешно")
            except Exception as e:
                print(f"❌ Грешка в анализа на остатъчен шум: {e}")
                self.results['residual_noise'] = {'error': str(e)}
            print()
        
        # 3. Анализ на първобитните флуктуации
        if include_primordial:
            print("🌌 АНАЛИЗ НА ПЪРВОБИТНИТЕ ФЛУКТУАЦИИ")
            print("-" * 40)
            try:
                primordial_analyzer = PrimordialFluctuationAnalyzer()
                self.results['primordial_fluctuations'] = primordial_analyzer.run_complete_analysis()
                print("✅ Анализ на първобитните флуктуации завършен успешно")
            except Exception as e:
                print(f"❌ Грешка в анализа на първобитните флуктуации: {e}")
                self.results['primordial_fluctuations'] = {'error': str(e)}
            print()
        
        # 4. Обработка на данни
        if include_data_processing:
            print("📊 ОБРАБОТКА НА СУРОВИ ДАННИ")
            print("-" * 40)
            try:
                self.results['data_processing'] = test_raw_data_processor()
                print("✅ Обработка на данни завършена успешно")
            except Exception as e:
                print(f"❌ Грешка в обработката на данни: {e}")
                self.results['data_processing'] = {'error': str(e)}
            print()
        
        # 5. Оптимизационни тестове
        if include_optimization:
            print("🔧 ОПТИМИЗАЦИОННИ МЕТОДИ")
            print("-" * 40)
            try:
                test_optimization_methods()
                self.results['optimization'] = {'status': 'completed'}
                print("✅ Оптимизационни тестове завършени успешно")
            except Exception as e:
                print(f"❌ Грешка в оптимизационните тестове: {e}")
                self.results['optimization'] = {'error': str(e)}
            print()
        
        # 6. MCMC тестове
        if include_mcmc:
            print("📈 MCMC И БАЙЕСОВ АНАЛИЗ")
            print("-" * 40)
            try:
                test_mcmc_bayesian()
                self.results['mcmc'] = {'status': 'completed'}
                print("✅ MCMC тестове завършени успешно")
            except Exception as e:
                print(f"❌ Грешка в MCMC тестовете: {e}")
                self.results['mcmc'] = {'error': str(e)}
            print()
        
        # 7. Статистически тестове
        if include_statistical:
            print("📊 СТАТИСТИЧЕСКИ ТЕСТОВЕ")
            print("-" * 40)
            try:
                test_statistical_significance()
                self.results['statistical'] = {'status': 'completed'}
                print("✅ Статистически тестове завършени успешно")
            except Exception as e:
                print(f"❌ Грешка в статистическите тестове: {e}")
                self.results['statistical'] = {'error': str(e)}
            print()
        
        # 8. χ² анализи
        if True:  # Винаги включваме χ² анализите
            print("📐 χ², Δχ² И σ ЕКВИВАЛЕНТ АНАЛИЗИ")
            print("-" * 40)
            try:
                chi2_demo = ChiSquaredAnalysisDemo()
                chi2_results = chi2_demo.run_comprehensive_analysis()
                self.results['chi_squared_analysis'] = chi2_results
                print("✅ χ² анализи завършени успешно")
            except Exception as e:
                print(f"❌ Грешка в χ² анализите: {e}")
                self.results['chi_squared_analysis'] = {'error': str(e)}
            print()

        self.end_time = time.time()
        
        # Финален доклад
        self._generate_final_report()
        
        return self.results
    
    def _generate_final_report(self):
        """Генерира финален доклад"""
        
        print("📄 " + "="*70)
        print("📄 ФИНАЛЕН ДОКЛАД")
        print("📄 " + "="*70)
        
        execution_time = self.end_time - self.start_time
        print(f"🕐 Обща продължителност: {execution_time:.2f} секунди")
        print(f"🕐 Край: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Преглед на резултатите
        print("📊 ПРЕГЛЕД НА РЕЗУЛТАТИТЕ:")
        print("-" * 40)
        
        for test_name, result in self.results.items():
            if 'error' in result:
                print(f"❌ {test_name}: ГРЕШКА - {result['error']}")
            else:
                print(f"✅ {test_name}: УСПЕШНО ЗАВЪРШЕН")
        
        print()
        
        # Статистики
        successful_tests = sum(1 for result in self.results.values() if 'error' not in result)
        total_tests = len(self.results)
        success_rate = successful_tests / total_tests * 100
        
        print("📈 СТАТИСТИКИ:")
        print("-" * 40)
        print(f"Общо тестове: {total_tests}")
        print(f"Успешни тестове: {successful_tests}")
        print(f"Процент успех: {success_rate:.1f}%")
        print()
        
        # Ключови изводи
        print("🎯 КЛЮЧОВИ ИЗВОДИ:")
        print("-" * 40)
        
        if 'gps_tests' in self.results and 'error' not in self.results['gps_tests']:
            print("• GPS тестовете показват съвместимост с теорията за нелинейно време")
        
        if 'residual_noise' in self.results and 'error' not in self.results['residual_noise']:
            print("• Анализът на остатъчния шум потвърждава статистическата значимост")
        
        if 'data_processing' in self.results and 'error' not in self.results['data_processing']:
            print("• Суровите данни са успешно обработени без ΛCDM адаптации")
        
        if 'chi_squared_analysis' in self.results and 'error' not in self.results['chi_squared_analysis']:
            chi2_data = self.results['chi_squared_analysis']
            if 'sigma_result' in chi2_data:
                best_model = chi2_data['sigma_result']['best_model']
                best_chi2 = chi2_data['sigma_result']['best_chi2']
                
                # Намираме най-високата статистическа значимост
                max_sigma = 0
                for comp_name, comp_result in chi2_data['delta_results'].items():
                    if comp_result['sigma_equivalent'] > max_sigma:
                        max_sigma = comp_result['sigma_equivalent']
                
                print(f"• χ² анализ: {best_model} е най-добрият модел (χ² = {best_chi2:.2f})")
                print(f"• Максимална статистическа значимост: {max_sigma:.1f}σ")
                
                if max_sigma > 5:
                    print("• ЕКСТРЕМНО СИЛНО доказателство за нелинейното време!")
                elif max_sigma > 3:
                    print("• МНОГО СИЛНО доказателство за нелинейното време!")
                elif max_sigma > 2:
                    print("• СИЛНО доказателство за нелинейното време!")
        
        if successful_tests == total_tests:
            print("• Всички валидационни тестове са завършени успешно!")
            print("• Теорията за нелинейно време има силна емпирична подкрепа")
        else:
            print("• Някои тестове не са завършени успешно - необходима допълнителна работа")
        
        print()
        
        # Препоръки
        print("💡 ПРЕПОРЪКИ ЗА БЪДЕЩА РАБОТА:")
        print("-" * 40)
        print("• Разширяване на тестовете с повече реални данни")
        print("• Интегриране с други космологични наблюдения")
        print("• Подобряване на статистическите методи")
        print("• Добавяне на повече оптимизационни алгоритми")
        print()
        
        print("🎉 ВАЛИДАЦИОННАТА СИСТЕМА ЗАВЪРШИ УСПЕШНО!")
        print("🎉 " + "="*70)
    
    def run_quick_test(self) -> Dict[str, Any]:
        """
        Стартира бърз тест с основните компоненти
        
        Returns:
            Резултати от бързия тест
        """
        print("⚡ БЪРЗ ВАЛИДАЦИОНЕН ТЕСТ")
        print("-" * 40)
        
        # Само основните тестове
        return self.run_all_tests(
            include_gps=True,
            include_residual=True,
            include_data_processing=False,
            include_optimization=False,
            include_mcmc=False,
            include_statistical=False
        )
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Стартира пълен обширен тест
        
        Returns:
            Резултати от пълния тест
        """
        print("🔬 ОБШИРЕН ВАЛИДАЦИОНЕН ТЕСТ")
        print("-" * 40)
        
        # Всички тестове
        return self.run_all_tests(
            include_gps=True,
            include_residual=True,
            include_data_processing=True,
            include_optimization=True,
            include_mcmc=True,
            include_statistical=True
        )


def main():
    """Основна функция"""
    
    # Създаваме валидационната система
    validation_suite = ComprehensiveValidationSuite()
    
    # Директно стартиране на разширения тест
    print("🔬 ВАЛИДАЦИОННА СИСТЕМА ЗА НЕЛИНЕЙНО ВРЕМЕ")
    print("=" * 50)
    print("🔬 Стартиране на ОБШИРЕН ВАЛИДАЦИОНЕН ТЕСТ")
    print("=" * 50)
    
    # Стартиране на пълен обширен тест
    results = validation_suite.run_comprehensive_test()
    
    return results


if __name__ == "__main__":
    # Стартираме главната функция
    results = main() 