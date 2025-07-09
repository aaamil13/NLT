"""
BAO анализатор с нелинейно време

Този модул имплементира анализ на барионните акустични осцилации (BAO)
в контекста на нелинейната времева космология. Сравнява теоретичните
предсказания с реални данни от BOSS, eBOSS и други проучвания.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import optimize, integrate
from typing import Dict, List, Tuple, Any, Optional
import logging

from common_utils.nonlinear_time_core import NonlinearTimeCosmology
from common_utils.cosmological_parameters import BAOData, PlanckCosmology
from common_utils.data_processing import BAODataProcessor, StatisticalAnalyzer

logger = logging.getLogger(__name__)

class BAOAnalyzer:
    """
    Анализатор на барионните акустични осцилации с нелинейно време
    
    Функционалности:
    - Сравнение с реални BAO данни
    - Изчисляване на D_V/r_s съотношения
    - Статистически анализ на съответствието
    - Оптимизация на параметрите
    """
    
    def __init__(self, nonlinear_params: Dict[str, float] = None):
        """
        Инициализация на BAO анализатора
        
        Args:
            nonlinear_params: Параметри за нелинейното време
        """
        # Използвай стандартните параметри ако не са предоставени
        if nonlinear_params is None:
            nonlinear_params = {
                'alpha': 1.5,
                'beta': 0.0,
                'gamma': 0.5,
                'delta': 0.1
            }
        
        # Инициализация на космологията
        self.cosmology = NonlinearTimeCosmology(**nonlinear_params)
        
        # Процесор за данни
        self.data_processor = BAODataProcessor()
        
        # Реални данни
        self.real_data = None
        self.processed_data = None
        
        logger.info("Инициализиран BAO анализатор с нелинейно време")
    
    def load_real_data(self, dataset: str = 'combined') -> Dict[str, np.ndarray]:
        """
        Зарежда реални BAO данни
        
        Args:
            dataset: Тип данни ('combined', 'boss', 'boss_dr12', 'eboss_dr16')
            
        Returns:
            Заредените данни
        """
        if dataset == 'combined':
            self.real_data = BAOData.get_combined_data()
        elif dataset == 'boss':
            self.real_data = BAOData.get_boss_only()
        elif dataset == 'boss_dr12':
            self.real_data = BAOData.BOSS_DR12
        elif dataset == 'eboss_dr16':
            self.real_data = BAOData.eBOSS_DR16
        else:
            raise ValueError(f"Неподдържан dataset: {dataset}")
        
        # Обработка на данните
        self.processed_data = self.data_processor.process_bao_measurements(
            self.real_data['z'],
            self.real_data['D_V_over_rs'],
            self.real_data['D_V_over_rs_err']
        )
        
        logger.info(f"Заредени {dataset} данни: {len(self.processed_data['z'])} точки")
        return self.processed_data
    
    def calculate_theoretical_dv_rs(self, z: np.ndarray, 
                                  r_s_reference: float = None) -> np.ndarray:
        """
        Изчислява теоретичните D_V/r_s стойности
        
        Args:
            z: Червени отмествания
            r_s_reference: Референтен звуков хоризонт (ако е None, използва се r_s от модела)
            
        Returns:
            Теоретичните D_V/r_s стойности
        """
        # Звуков хоризонт
        if r_s_reference is None:
            r_s = self.cosmology.sound_horizon_integral()
        else:
            r_s = r_s_reference
        
        # Обемно усреднено разстояние
        D_V = self.cosmology.volume_averaged_distance(z)
        
        # Съотношение D_V/r_s
        D_V_over_rs = D_V / r_s
        
        return D_V_over_rs
    
    def compare_with_observations(self, z_obs: np.ndarray = None, 
                                D_V_obs: np.ndarray = None, 
                                errors_obs: np.ndarray = None) -> Dict[str, Any]:
        """
        Сравнява теоретичните предсказания с наблюденията
        
        Args:
            z_obs: Наблюдавани червени отмествания
            D_V_obs: Наблюдавани D_V/r_s стойности
            errors_obs: Грешки в наблюденията
            
        Returns:
            Резултати от сравнението
        """
        # Използвай заредените данни ако не са предоставени
        if z_obs is None:
            if self.processed_data is None:
                self.load_real_data()
            z_obs = self.processed_data['z']
            D_V_obs = self.processed_data['D_V_over_rs']
            errors_obs = self.processed_data['errors']
        
        # Теоретични предсказания
        D_V_theory = self.calculate_theoretical_dv_rs(z_obs)
        
        # Статистически анализ
        stats = StatisticalAnalyzer.goodness_of_fit_summary(
            D_V_theory, D_V_obs, errors_obs, n_params=4
        )
        
        # Резидуали
        residuals = (D_V_theory - D_V_obs) / errors_obs
        
        # Дополнителни статистики
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        max_residual = np.max(np.abs(residuals))
        
        logger.info(f"χ²/dof = {stats['reduced_chi_squared']:.2f}")
        logger.info(f"Средно отклонение: {mean_residual:.3f} σ")
        logger.info(f"Стандартно отклонение: {std_residual:.3f} σ")
        
        return {
            'z_obs': z_obs,
            'D_V_obs': D_V_obs,
            'D_V_theory': D_V_theory,
            'errors_obs': errors_obs,
            'residuals': residuals,
            'statistics': stats,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'max_residual': max_residual,
            'agreement_level': self._assess_agreement_level(stats['reduced_chi_squared'])
        }
    
    def _assess_agreement_level(self, reduced_chi_squared: float) -> str:
        """
        Оценява нивото на съответствие на базата на χ²/dof
        
        Args:
            reduced_chi_squared: Редуциран χ²
            
        Returns:
            Текстова оценка на съответствието
        """
        if reduced_chi_squared <= 1.2:
            return "Отлично съответствие"
        elif reduced_chi_squared <= 2.0:
            return "Добро съответствие"
        elif reduced_chi_squared <= 3.0:
            return "Приемливо съответствие"
        elif reduced_chi_squared <= 5.0:
            return "Слабо съответствие"
        else:
            return "Неприемливо съответствие"
    
    def sensitivity_analysis(self, parameter: str, 
                           param_range: Tuple[float, float],
                           n_steps: int = 20) -> Dict[str, np.ndarray]:
        """
        Анализ на чувствителността спрямо параметрите
        
        Args:
            parameter: Име на параметъра за варииране
            param_range: Диапазон на параметъра
            n_steps: Брой стъпки в диапазона
            
        Returns:
            Резултати от анализа на чувствителността
        """
        if self.processed_data is None:
            self.load_real_data()
        
        # Параметрична мрежа
        param_values = np.linspace(param_range[0], param_range[1], n_steps)
        chi_squared_values = np.zeros(n_steps)
        
        # Оригинални параметри
        original_params = {
            'alpha': self.cosmology.alpha,
            'beta': self.cosmology.beta,
            'gamma': self.cosmology.gamma,
            'delta': self.cosmology.delta
        }
        
        # Варииране на параметъра
        for i, param_val in enumerate(param_values):
            # Временна космология с новия параметър
            temp_params = original_params.copy()
            temp_params[parameter] = param_val
            
            temp_cosmology = NonlinearTimeCosmology(**temp_params)
            
            # Теоретични предсказания
            D_V_theory = temp_cosmology.volume_averaged_distance(self.processed_data['z']) / temp_cosmology.sound_horizon_integral()
            
            # χ² стойност
            chi_squared_values[i] = StatisticalAnalyzer.calculate_chi_squared(
                D_V_theory, self.processed_data['D_V_over_rs'], self.processed_data['errors']
            )
        
        # Намери най-добрата стойност
        best_idx = np.argmin(chi_squared_values)
        best_param = param_values[best_idx]
        best_chi_squared = chi_squared_values[best_idx]
        
        logger.info(f"Най-добра стойност за {parameter}: {best_param:.3f}")
        logger.info(f"Минимален χ²: {best_chi_squared:.2f}")
        
        return {
            'parameter': parameter,
            'param_values': param_values,
            'chi_squared_values': chi_squared_values,
            'best_param': best_param,
            'best_chi_squared': best_chi_squared,
            'original_param': original_params[parameter]
        }
    
    def z_evolution_analysis(self, z_min: float = 0.01, z_max: float = 2.0, 
                           n_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Анализ на еволюцията на D_V/r_s с червеното отместване
        
        Args:
            z_min: Минимално червено отместване
            z_max: Максимално червено отместване
            n_points: Брой точки в мрежата
            
        Returns:
            Резултати от еволюционния анализ
        """
        # Логаритмична мрежа
        z_grid = np.logspace(np.log10(z_min), np.log10(z_max), n_points)
        
        # Теоретични предсказания
        D_V_theory = self.calculate_theoretical_dv_rs(z_grid)
        
        # Стандартна ΛCDM за сравнение
        lambda_cdm = NonlinearTimeCosmology(alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
        D_V_lambda_cdm = lambda_cdm.volume_averaged_distance(z_grid) / lambda_cdm.sound_horizon_integral()
        
        # Относителна разлика
        relative_diff = (D_V_theory - D_V_lambda_cdm) / D_V_lambda_cdm * 100
        
        logger.info(f"Максимална разлика от ΛCDM: {np.max(np.abs(relative_diff)):.2f}%")
        
        return {
            'z_grid': z_grid,
            'D_V_nonlinear': D_V_theory,
            'D_V_lambda_cdm': D_V_lambda_cdm,
            'relative_difference': relative_diff,
            'max_difference': np.max(np.abs(relative_diff))
        }
    
    def parameter_correlation_analysis(self) -> Dict[str, np.ndarray]:
        """
        Анализ на корелациите между параметрите
        
        Returns:
            Корелационна матрица и свързани статистики
        """
        if self.processed_data is None:
            self.load_real_data()
        
        # Параметри за тестване
        param_names = ['alpha', 'beta', 'gamma', 'delta']
        n_params = len(param_names)
        
        # Мрежа от параметрични стойности
        param_grids = {
            'alpha': np.linspace(0.5, 2.5, 5),
            'beta': np.linspace(-0.2, 0.2, 3),
            'gamma': np.linspace(0.2, 0.8, 4),
            'delta': np.linspace(0.05, 0.15, 3)
        }
        
        # Изчисляване на χ² за всички комбинации
        chi_squared_grid = []
        param_combinations = []
        
        for alpha in param_grids['alpha']:
            for beta in param_grids['beta']:
                for gamma in param_grids['gamma']:
                    for delta in param_grids['delta']:
                        # Временна космология
                        temp_cosmology = NonlinearTimeCosmology(
                            alpha=alpha, beta=beta, gamma=gamma, delta=delta
                        )
                        
                        # Теоретични предсказания
                        D_V_theory = temp_cosmology.volume_averaged_distance(self.processed_data['z']) / temp_cosmology.sound_horizon_integral()
                        
                        # χ² стойност
                        chi_squared = StatisticalAnalyzer.calculate_chi_squared(
                            D_V_theory, self.processed_data['D_V_over_rs'], self.processed_data['errors']
                        )
                        
                        chi_squared_grid.append(chi_squared)
                        param_combinations.append([alpha, beta, gamma, delta])
        
        # Конвертиране в масиви
        chi_squared_grid = np.array(chi_squared_grid)
        param_combinations = np.array(param_combinations)
        
        # Корелационна матрица
        correlation_matrix = np.corrcoef(param_combinations.T)
        
        # Най-добра комбинация
        best_idx = np.argmin(chi_squared_grid)
        best_params = param_combinations[best_idx]
        best_chi_squared = chi_squared_grid[best_idx]
        
        logger.info(f"Най-добра комбинация: α={best_params[0]:.2f}, β={best_params[1]:.2f}, γ={best_params[2]:.2f}, δ={best_params[3]:.2f}")
        logger.info(f"Минимален χ²: {best_chi_squared:.2f}")
        
        return {
            'param_names': param_names,
            'correlation_matrix': correlation_matrix,
            'param_combinations': param_combinations,
            'chi_squared_grid': chi_squared_grid,
            'best_params': best_params,
            'best_chi_squared': best_chi_squared
        }
    
    def comprehensive_analysis_report(self) -> Dict[str, Any]:
        """
        Генерира обширен доклад за BAO анализа
        
        Returns:
            Пълен доклад с всички анализи
        """
        logger.info("🔍 Започва обширен BAO анализ...")
        
        # Зареждане на данни
        self.load_real_data()
        
        # Основно сравнение
        comparison = self.compare_with_observations()
        
        # Анализ на чувствителността
        alpha_sensitivity = self.sensitivity_analysis('alpha', (0.5, 2.5))
        gamma_sensitivity = self.sensitivity_analysis('gamma', (0.2, 0.8))
        
        # Еволюционен анализ
        evolution = self.z_evolution_analysis()
        
        # Корелационен анализ
        correlation = self.parameter_correlation_analysis()
        
        # Обобщение
        report = {
            'comparison_results': comparison,
            'sensitivity_analysis': {
                'alpha': alpha_sensitivity,
                'gamma': gamma_sensitivity
            },
            'evolution_analysis': evolution,
            'correlation_analysis': correlation,
            'model_parameters': {
                'alpha': self.cosmology.alpha,
                'beta': self.cosmology.beta,
                'gamma': self.cosmology.gamma,
                'delta': self.cosmology.delta,
                'H0': self.cosmology.H0,
                'Omega_m': self.cosmology.Omega_m,
                'Omega_Lambda': self.cosmology.Omega_Lambda
            },
            'data_statistics': self.processed_data['statistics']
        }
        
        logger.info("✅ Обширният BAO анализ е завършен!")
        return report


def test_bao_analyzer():
    """Тест на BAO анализатора"""
    print("🧪 ТЕСТ НА BAO АНАЛИЗАТОРА")
    print("=" * 50)
    
    # Създаване на анализатор
    analyzer = BAOAnalyzer()
    
    # Зареждане на данни
    print("\n📊 Зареждане на реални данни...")
    analyzer.load_real_data('combined')
    print(f"Заредени {len(analyzer.processed_data['z'])} точки")
    
    # Основно сравнение
    print("\n🔍 Сравнение с наблюденията...")
    comparison = analyzer.compare_with_observations()
    print(f"χ²/dof = {comparison['statistics']['reduced_chi_squared']:.2f}")
    print(f"Ниво на съответствие: {comparison['agreement_level']}")
    
    # Анализ на чувствителността
    print("\n📈 Анализ на чувствителността...")
    sensitivity = analyzer.sensitivity_analysis('alpha', (1.0, 2.0), n_steps=10)
    print(f"Най-добра стойност за α: {sensitivity['best_param']:.3f}")
    
    # Еволюционен анализ
    print("\n🌌 Еволюционен анализ...")
    evolution = analyzer.z_evolution_analysis(n_points=50)
    print(f"Максимална разлика от ΛCDM: {evolution['max_difference']:.2f}%")
    
    print("\n✅ Всички тестове завършиха успешно!")
    
    return analyzer


if __name__ == "__main__":
    analyzer = test_bao_analyzer() 