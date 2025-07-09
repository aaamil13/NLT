"""
GPS времева дилатация тестове
============================

Тестване на теорията за нелинейно време чрез GPS данни.
Използваме различни оптимизационни методи и статистически тестове.

Автор: Система за анализ на нелинейно време
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, G
from scipy.optimize import minimize
import warnings
from typing import Dict, List, Tuple, Any, Optional

# Импортираме общите утилити
from validation_tests.common_utils.optimization_engines import DifferentialEvolutionOptimizer, BasinhoppingOptimizer, HybridOptimizer
from validation_tests.common_utils.mcmc_bayesian import MCMCBayesianAnalyzer, BayesianModelComparison
from validation_tests.common_utils.statistical_tests import StatisticalSignificanceTest, CrossValidationAnalysis
from validation_tests.common_utils.data_processors import RawDataProcessor

# Импортираме нашите модели
from lib.advanced_analytical_functions import AdvancedAnalyticalFunctions

# Настройка на логирането
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Модулно ниво функции за оптимизация
def objective_classical_global(params, models, observed_data, gps_data):
    """Глобална функция за класическия модел"""
    predicted = models['classical_model'](params, gps_data)
    return np.sum((observed_data - predicted)**2)

def objective_nonlinear_global(params, models, observed_data, gps_data):
    """Глобална функция за нелинейния модел"""
    predicted = models['nonlinear_model'](params, gps_data)
    return np.sum((observed_data - predicted)**2)

warnings.filterwarnings('ignore')

# Константи за GPS
GPS_ALTITUDE = 20200e3  # м (средна височина на GPS сателитите)
EARTH_RADIUS = 6.371e6  # м
GPS_ORBITAL_VELOCITY = 3874.0  # m/s
GPS_ORBITAL_PERIOD = 43200.0  # s (12 часа)
GPS_FREQUENCY = 1.57542e9  # Hz (L1 честота)

# Физически константи (натурални единици)
G_NEWTON = 6.67e-11  # m³/kg/s²
C_LIGHT = 3e8  # m/s
EARTH_MASS = 5.972e24  # kg


class GPSTimeDilationTest:
    """
    Клас за тестване на времева дилатация в GPS системи
    """
    
    def __init__(self, use_nonlinear_time: bool = True):
        """
        Инициализация на GPS теста
        
        Args:
            use_nonlinear_time: Дали да използва нелинейното време
        """
        self.use_nonlinear_time = use_nonlinear_time
        self.aaf = AdvancedAnalyticalFunctions()
        self.gps_data = {}
        self.test_results = {}
        
    def generate_synthetic_gps_data(self, n_satellites: int = 24, 
                                   time_duration: float = 86400.0,
                                   noise_level: float = 1e-12) -> Dict[str, Any]:
        """
        Генерира синтетични GPS данни
        
        Args:
            n_satellites: Брой сателити
            time_duration: Продължителност на наблюдението (секунди)
            noise_level: Ниво на шума
            
        Returns:
            Синтетични GPS данни
        """
        # Време
        t = np.linspace(0, time_duration, 1000)
        
        data = {
            'time': t,
            'satellites': {},
            'n_satellites': n_satellites,
            'noise_level': noise_level
        }
        
        for sat_id in range(n_satellites):
            # Орбитални параметри
            phase = (2 * np.pi * sat_id / n_satellites)  # Фазово отместване
            
            # Позиция на сателита
            x_sat = GPS_ALTITUDE * np.cos(2 * np.pi * t / GPS_ORBITAL_PERIOD + phase)
            y_sat = GPS_ALTITUDE * np.sin(2 * np.pi * t / GPS_ORBITAL_PERIOD + phase)
            z_sat = np.zeros_like(t)  # Приблизително кръгова орбита
            
            # Скорост на сателита
            v_sat = GPS_ORBITAL_VELOCITY * np.ones_like(t)
            
            # Гравитационен потенциал
            r_sat = np.sqrt(x_sat**2 + y_sat**2 + z_sat**2)
            gravitational_potential = -G_NEWTON * EARTH_MASS / r_sat  # Земна маса
            
            # Класическа времева дилатация (Einstein)
            sr_dilation = -v_sat**2 / (2 * C_LIGHT**2)  # Специална релативност
            gr_dilation = gravitational_potential / C_LIGHT**2  # Обща релативност
            
            # Общо отместване на времето (класическо)
            classical_time_offset = sr_dilation + gr_dilation
            
            # Нелинейно време отместване (нашия модел)
            if self.use_nonlinear_time:
                # Приблизително z за GPS времева скала
                z_gps = 1e-10  # Много малко червено отместване
                nonlinear_correction = self.aaf.analytical_t_z_approximation(z_gps)
                nonlinear_time_offset = classical_time_offset * (1 + nonlinear_correction)
            else:
                nonlinear_time_offset = classical_time_offset
            
            # Додаваме шум
            noise = np.random.normal(0, noise_level, len(t))
            
            data['satellites'][sat_id] = {
                'position': np.column_stack([x_sat, y_sat, z_sat]),
                'velocity': v_sat,
                'gravitational_potential': gravitational_potential,
                'classical_time_offset': classical_time_offset,
                'nonlinear_time_offset': nonlinear_time_offset,
                'observed_time_offset': nonlinear_time_offset + noise,
                'noise': noise
            }
        
        self.gps_data = data
        return data
    
    def define_model_functions(self) -> Dict[str, Any]:
        """
        Дефинира модели за сравнение
        
        Returns:
            Речник с модели
        """
        def classical_model(params, data):
            """Класически модел за времева дилатация"""
            a, b = params  # Коефициенти за SR и GR
            
            predicted_offsets = []
            for sat_id in range(data['n_satellites']):
                sat_data = data['satellites'][sat_id]
                
                v_sat = sat_data['velocity']
                phi_sat = sat_data['gravitational_potential']
                
                # Класическа времева дилатация
                sr_term = a * (-v_sat**2 / (2 * C_LIGHT**2))
                gr_term = b * (phi_sat / C_LIGHT**2)
                
                predicted_offset = sr_term + gr_term
                predicted_offsets.append(predicted_offset)
            
            return np.concatenate(predicted_offsets)
        
        def nonlinear_model(params, data):
            """Нелинейно време модел"""
            a, b, gamma = params  # Коефициенти за SR, GR и нелинейност
            
            predicted_offsets = []
            for sat_id in range(data['n_satellites']):
                sat_data = data['satellites'][sat_id]
                
                v_sat = sat_data['velocity']
                phi_sat = sat_data['gravitational_potential']
                
                # Класическа времева дилатация
                sr_term = a * (-v_sat**2 / (2 * C_LIGHT**2))
                gr_term = b * (phi_sat / C_LIGHT**2)
                classical_offset = sr_term + gr_term
                
                # Нелинейна корекция
                z_equivalent = gamma * np.abs(classical_offset)
                nonlinear_correction = self.aaf.analytical_t_z_approximation(z_equivalent)
                
                predicted_offset = classical_offset * (1 + nonlinear_correction)
                predicted_offsets.append(predicted_offset)
            
            return np.concatenate(predicted_offsets)
        
        # Обединяваме наблюдаваните данни
        observed_data = []
        for sat_id in range(self.gps_data['n_satellites']):
            sat_data = self.gps_data['satellites'][sat_id]
            observed_data.append(sat_data['observed_time_offset'])
        observed_data = np.concatenate(observed_data)
        
        return {
            'classical_model': classical_model,
            'nonlinear_model': nonlinear_model,
            'observed_data': observed_data
        }
    
    def run_optimization_tests(self) -> Dict[str, Any]:
        """
        Стартира оптимизационни тестове с различни методи
        
        Returns:
            Резултати от оптимизацията
        """
        models = self.define_model_functions()
        observed_data = models['observed_data']
        
        results = {}
        
        # Тестваме класическия модел
        print("Тестване на класическия модел...")
        
        # Differential Evolution
        de_optimizer = DifferentialEvolutionOptimizer(max_iterations=200, parallel=False)  # Изключваме паралелизма
        de_result_classical = de_optimizer.optimize(
            lambda params: objective_classical_global(params, models, observed_data, self.gps_data),
            [(0.5, 1.5), (0.5, 1.5)],  # Граници около теоретичните стойности
            ()
        )
        
        # Basinhopping
        bh_optimizer = BasinhoppingOptimizer(n_iterations=100)
        bh_result_classical = bh_optimizer.optimize(
            lambda params: objective_classical_global(params, models, observed_data, self.gps_data),
            np.array([1.0, 1.0]),  # Начална оценка
            [(0.5, 1.5), (0.5, 1.5)]
        )
        
        results['classical_model'] = {
            'differential_evolution': de_result_classical,
            'basinhopping': bh_result_classical
        }
        
        # Тестваме нелинейния модел
        print("Тестване на нелинейния модел...")
        
        # Differential Evolution
        de_optimizer_nl = DifferentialEvolutionOptimizer(max_iterations=200, parallel=False)
        de_result_nonlinear = de_optimizer_nl.optimize(
            lambda params: objective_nonlinear_global(params, models, observed_data, self.gps_data),
            [(0.5, 1.5), (0.5, 1.5), (1e-12, 1e-8)],  # Граници за gamma
            ()
        )
        
        # Basinhopping
        bh_optimizer_nl = BasinhoppingOptimizer(n_iterations=100)
        bh_result_nonlinear = bh_optimizer_nl.optimize(
            lambda params: objective_nonlinear_global(params, models, observed_data, self.gps_data),
            np.array([1.0, 1.0, 1e-10]),  # Начална оценка
            [(0.5, 1.5), (0.5, 1.5), (1e-12, 1e-8)]
        )
        
        results['nonlinear_model'] = {
            'differential_evolution': de_result_nonlinear,
            'basinhopping': bh_result_nonlinear
        }
        
        self.test_results['optimization'] = results
        return results
    
    def run_mcmc_analysis(self) -> Dict[str, Any]:
        """
        Стартира MCMC анализ за модел сравнение
        
        Returns:
            Резултати от MCMC
        """
        models = self.define_model_functions()
        observed_data = models['observed_data']
        n_data = len(observed_data)
        
        # Дефинираме log-likelihood функции
        def log_likelihood_classical(params, data):
            a, b, sigma = params
            predicted = models['classical_model']([a, b], self.gps_data)
            return -0.5 * np.sum((data - predicted)**2 / sigma**2 + np.log(2 * np.pi * sigma**2))
        
        def log_likelihood_nonlinear(params, data):
            a, b, gamma, sigma = params
            predicted = models['nonlinear_model']([a, b, gamma], self.gps_data)
            return -0.5 * np.sum((data - predicted)**2 / sigma**2 + np.log(2 * np.pi * sigma**2))
        
        # Дефинираме prior функции
        def log_prior_classical(params):
            a, b, sigma = params
            if 0.5 < a < 1.5 and 0.5 < b < 1.5 and 1e-15 < sigma < 1e-10:
                return 0.0
            return -np.inf
        
        def log_prior_nonlinear(params):
            a, b, gamma, sigma = params
            if 0.5 < a < 1.5 and 0.5 < b < 1.5 and 1e-12 < gamma < 1e-8 and 1e-15 < sigma < 1e-10:
                return 0.0
            return -np.inf
        
        # Байесово сравнение на модели
        comparison = BayesianModelComparison()
        
        # Добавяме модели
        comparison.add_model(
            'classical',
            log_likelihood_classical,
            log_prior_classical,
            [(0.5, 1.5), (0.5, 1.5), (1e-15, 1e-10)],
            np.array([1.0, 1.0, 1e-12])
        )
        
        comparison.add_model(
            'nonlinear',
            log_likelihood_nonlinear,
            log_prior_nonlinear,
            [(0.5, 1.5), (0.5, 1.5), (1e-12, 1e-8), (1e-15, 1e-10)],
            np.array([1.0, 1.0, 1e-10, 1e-12])
        )
        
        # Стартираме сравнението
        mcmc_results = comparison.run_comparison(
            observed_data,
            {'n_walkers': 50, 'n_steps': 1000, 'n_burn': 200}
        )
        
        self.test_results['mcmc'] = mcmc_results
        return mcmc_results
    
    def run_statistical_tests(self) -> Dict[str, Any]:
        """
        Стартира статистически тестове
        
        Returns:
            Резултати от статистическите тестове
        """
        if 'optimization' not in self.test_results:
            print("Първо трябва да се стартират оптимизационните тестове")
            return {}
        
        models = self.define_model_functions()
        observed_data = models['observed_data']
        
        # Получаваме най-добрите параметри
        opt_results = self.test_results['optimization']
        best_classical = opt_results['classical_model']['differential_evolution']['best_parameters']
        best_nonlinear = opt_results['nonlinear_model']['differential_evolution']['best_parameters']
        
        # Пресмятаме остатъци
        predicted_classical = models['classical_model'](best_classical, self.gps_data)
        predicted_nonlinear = models['nonlinear_model'](best_nonlinear, self.gps_data)
        
        residuals_classical = observed_data - predicted_classical
        residuals_nonlinear = observed_data - predicted_nonlinear
        
        # Статистически тестове
        stat_test = StatisticalSignificanceTest()
        
        # Анализ на остатъци за класическия модел
        classical_analysis = stat_test.comprehensive_residual_analysis(
            residuals_classical, predicted_classical
        )
        
        # Анализ на остатъци за нелинейния модел
        nonlinear_analysis = stat_test.comprehensive_residual_analysis(
            residuals_nonlinear, predicted_nonlinear
        )
        
        # F-тест за сравнение на модели
        rss_classical = np.sum(residuals_classical**2)
        rss_nonlinear = np.sum(residuals_nonlinear**2)
        
        f_test = stat_test.f_test_model_comparison(
            rss_classical, rss_nonlinear,
            len(best_classical), len(best_nonlinear),
            len(observed_data)
        )
        
        results = {
            'classical_residuals_analysis': classical_analysis,
            'nonlinear_residuals_analysis': nonlinear_analysis,
            'f_test': f_test,
            'residuals_classical': residuals_classical,
            'residuals_nonlinear': residuals_nonlinear
        }
        
        self.test_results['statistical'] = results
        return results
    
    def plot_results(self, save_path: str = None):
        """
        Създава графики с резултатите
        
        Args:
            save_path: Път за записване
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. GPS данни
        if self.gps_data:
            time = self.gps_data['time']
            
            # Показваме данни от първия сателит
            sat_0 = self.gps_data['satellites'][0]
            
            axes[0, 0].plot(time/3600, sat_0['classical_time_offset']*1e12, 
                           'b-', label='Класическо време')
            axes[0, 0].plot(time/3600, sat_0['nonlinear_time_offset']*1e12, 
                           'r-', label='Нелинейно време')
            axes[0, 0].scatter(time[::100]/3600, sat_0['observed_time_offset'][::100]*1e12, 
                              alpha=0.6, s=10, label='Наблюдения')
            
            axes[0, 0].set_xlabel('Време [h]')
            axes[0, 0].set_ylabel('Времево отместване [ps]')
            axes[0, 0].set_title('GPS времева дилатация')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Резултати от оптимизацията
        if 'optimization' in self.test_results:
            opt_results = self.test_results['optimization']
            
            methods = ['differential_evolution', 'basinhopping']
            models = ['classical_model', 'nonlinear_model']
            
            scores = np.zeros((len(models), len(methods)))
            
            for i, model in enumerate(models):
                for j, method in enumerate(methods):
                    scores[i, j] = opt_results[model][method]['best_score']
            
            im = axes[0, 1].imshow(scores, cmap='viridis', aspect='auto')
            axes[0, 1].set_xticks(range(len(methods)))
            axes[0, 1].set_xticklabels(methods, rotation=45)
            axes[0, 1].set_yticks(range(len(models)))
            axes[0, 1].set_yticklabels(models)
            axes[0, 1].set_title('Оптимизационни резултати')
            plt.colorbar(im, ax=axes[0, 1])
        
        # 3. MCMC резултати
        if 'mcmc' in self.test_results:
            mcmc_results = self.test_results['mcmc']
            comparison = mcmc_results['comparison']
            
            criteria = ['AIC', 'BIC', 'DIC', 'WAIC']
            classical_scores = [comparison[c]['values']['classical'] for c in criteria]
            nonlinear_scores = [comparison[c]['values']['nonlinear'] for c in criteria]
            
            x = np.arange(len(criteria))
            width = 0.35
            
            axes[0, 2].bar(x - width/2, classical_scores, width, label='Класически')
            axes[0, 2].bar(x + width/2, nonlinear_scores, width, label='Нелинеен')
            axes[0, 2].set_xlabel('Критерий')
            axes[0, 2].set_ylabel('Стойност')
            axes[0, 2].set_title('Байесово сравнение')
            axes[0, 2].set_xticks(x)
            axes[0, 2].set_xticklabels(criteria)
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Остатъци за класическия модел
        if 'statistical' in self.test_results:
            stat_results = self.test_results['statistical']
            
            residuals_classical = stat_results['residuals_classical']
            residuals_nonlinear = stat_results['residuals_nonlinear']
            
            axes[1, 0].hist(residuals_classical*1e12, bins=30, alpha=0.7, 
                           label='Класически', density=True)
            axes[1, 0].set_xlabel('Остатъци [ps]')
            axes[1, 0].set_ylabel('Плътност')
            axes[1, 0].set_title('Остатъци - Класически модел')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Остатъци за нелинейния модел
            axes[1, 1].hist(residuals_nonlinear*1e12, bins=30, alpha=0.7, 
                           label='Нелинеен', density=True)
            axes[1, 1].set_xlabel('Остатъци [ps]')
            axes[1, 1].set_ylabel('Плътност')
            axes[1, 1].set_title('Остатъци - Нелинеен модел')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Сравнение на остатъци
            axes[1, 2].scatter(residuals_classical*1e12, residuals_nonlinear*1e12, 
                              alpha=0.6, s=10)
            axes[1, 2].plot([-100, 100], [-100, 100], 'r--', alpha=0.7)
            axes[1, 2].set_xlabel('Остатъци класически [ps]')
            axes[1, 2].set_ylabel('Остатъци нелинеен [ps]')
            axes[1, 2].set_title('Сравнение на остатъци')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Генерира подробен доклад
        
        Returns:
            Текстов доклад
        """
        report = []
        report.append("=" * 80)
        report.append("GPS ВРЕМЕВА ДИЛАТАЦИЯ ТЕСТ")
        report.append("=" * 80)
        report.append("")
        
        # Основни параметри
        report.append("ПАРАМЕТРИ НА ТЕСТА:")
        report.append("-" * 30)
        report.append(f"Използва нелинейно време: {self.use_nonlinear_time}")
        if self.gps_data:
            report.append(f"Брой сателити: {self.gps_data['n_satellites']}")
            report.append(f"Ниво на шум: {self.gps_data['noise_level']}")
        report.append("")
        
        # Оптимизационни резултати
        if 'optimization' in self.test_results:
            report.append("ОПТИМИЗАЦИОННИ РЕЗУЛТАТИ:")
            report.append("-" * 30)
            
            opt_results = self.test_results['optimization']
            
            for model_name, results in opt_results.items():
                report.append(f"\n{model_name.upper()}:")
                
                for method_name, result in results.items():
                    report.append(f"  {method_name}:")
                    report.append(f"    Най-добра стойност: {result['best_score']:.2e}")
                    report.append(f"    Параметри: {result['best_parameters']}")
                    report.append(f"    Време: {result['execution_time']:.2f}s")
            
            report.append("")
        
        # MCMC резултати
        if 'mcmc' in self.test_results:
            report.append("MCMC РЕЗУЛТАТИ:")
            report.append("-" * 30)
            
            mcmc_results = self.test_results['mcmc']
            comparison = mcmc_results['comparison']
            
            for criterion in ['AIC', 'BIC', 'DIC', 'WAIC']:
                best_model = comparison[criterion]['best_model']
                report.append(f"{criterion}: Най-добър модел - {best_model}")
            
            report.append("")
        
        # Статистически тестове
        if 'statistical' in self.test_results:
            report.append("СТАТИСТИЧЕСКИ ТЕСТОВЕ:")
            report.append("-" * 30)
            
            stat_results = self.test_results['statistical']
            
            # F-тест
            f_test = stat_results['f_test']
            report.append(f"F-тест:")
            report.append(f"  F-статистика: {f_test['f_statistic']:.4f}")
            report.append(f"  p-стойност: {f_test['p_value']:.6f}")
            report.append(f"  Заключение: {f_test['interpretation']}")
            
            report.append("")
        
        return "\n".join(report)
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Стартира пълен тест
        
        Returns:
            Всички резултати
        """
        print("🚀 Стартиране на GPS времева дилатация тест...")
        
        # 1. Генериране на данни
        print("📊 Генериране на GPS данни...")
        self.generate_synthetic_gps_data()
        
        # 2. Оптимизационни тестове
        print("🔍 Оптимизационни тестове...")
        self.run_optimization_tests()
        
        # 3. MCMC анализ
        print("📈 MCMC анализ...")
        self.run_mcmc_analysis()
        
        # 4. Статистически тестове
        print("📋 Статистически тестове...")
        self.run_statistical_tests()
        
        # 5. Създаване на графики
        print("📈 Създаване на графики...")
        self.plot_results()
        
        # 6. Генериране на доклад
        print("📄 Генериране на доклад...")
        report = self.generate_report()
        print(report)
        
        print("✅ GPS тестът завърши успешно!")
        
        return {
            'gps_data': self.gps_data,
            'test_results': self.test_results,
            'report': report
        }


def test_gps_time_dilation():
    """
    Тестова функция за GPS времева дилатация
    """
    # Тест с класическо време
    print("Тест с класическо време:")
    classical_test = GPSTimeDilationTest(use_nonlinear_time=False)
    classical_results = classical_test.run_comprehensive_test()
    
    print("\n" + "="*80 + "\n")
    
    # Тест с нелинейно време
    print("Тест с нелинейно време:")
    nonlinear_test = GPSTimeDilationTest(use_nonlinear_time=True)
    nonlinear_results = nonlinear_test.run_comprehensive_test()
    
    return {
        'classical': classical_results,
        'nonlinear': nonlinear_results
    }


if __name__ == "__main__":
    results = test_gps_time_dilation() 