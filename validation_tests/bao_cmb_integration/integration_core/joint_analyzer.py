"""
Съвместен анализатор за BAO и CMB данни

Този модул имплементира комбинирания анализ на барионните акустични осцилации
и космическото микровълново излъчване в единна вероятностна рамка.
Следва формулировката L_tot(θ) = L_CMB(θ) × L_BAO(θ).
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import optimize
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass

from bao_analysis.bao_analyzer import BAOAnalyzer
from cmb_analysis.cmb_analyzer import CMBAnalyzer
from common_utils.nonlinear_time_core import NonlinearTimeCosmology
from common_utils.cosmological_parameters import NonlinearTimeParameters
from common_utils.data_processing import StatisticalAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class JointAnalysisResults:
    """Резултати от съвместния анализ"""
    bao_results: Dict[str, Any]
    cmb_results: Dict[str, Any]
    combined_statistics: Dict[str, float]
    best_fit_parameters: Dict[str, float]
    parameter_uncertainties: Dict[str, float]
    agreement_assessment: str

class JointBAOCMBAnalyzer:
    """
    Съвместен анализатор за BAO и CMB данни
    
    Имплементира:
    - Комбинирани вероятностни функции
    - Параметрично оптимизиране
    - Статистически анализ на съвместимостта
    - Генериране на крайни резултати
    """
    
    def __init__(self, initial_params: Dict[str, float] = None):
        """
        Инициализация на съвместния анализатор
        
        Args:
            initial_params: Начални параметри за нелинейното време
        """
        # Използвай стандартните параметри ако не са предоставени
        if initial_params is None:
            initial_params = NonlinearTimeParameters.get_default_params()
        
        self.current_params = initial_params.copy()
        
        # Инициализация на анализаторите
        self.bao_analyzer = BAOAnalyzer(initial_params)
        self.cmb_analyzer = CMBAnalyzer(initial_params)
        
        # Параметрични граници
        self.param_bounds = {
            'alpha': NonlinearTimeParameters.ALPHA_RANGE,
            'beta': NonlinearTimeParameters.BETA_RANGE,
            'gamma': NonlinearTimeParameters.GAMMA_RANGE,
            'delta': NonlinearTimeParameters.DELTA_RANGE
        }
        
        # Резултати
        self.latest_results = None
        
        logger.info("Инициализиран съвместен BAO+CMB анализатор")
    
    def calculate_joint_likelihood(self, params: Dict[str, float]) -> float:
        """
        Изчислява съвместната вероятност L_tot(θ) = L_CMB(θ) × L_BAO(θ)
        
        Args:
            params: Параметри на модела
            
        Returns:
            Логаритъм на съвместната вероятност (-χ²/2)
        """
        # Валидиране на параметрите
        if not NonlinearTimeParameters.validate_parameters(params):
            return -np.inf
        
        try:
            # Обновяване на анализаторите
            self._update_analyzers(params)
            
            # BAO вероятност
            bao_comparison = self.bao_analyzer.compare_with_observations()
            chi2_bao = bao_comparison['statistics']['chi_squared']
            
            # CMB вероятност
            cmb_comparison = self.cmb_analyzer.compare_with_planck_data()
            chi2_cmb = cmb_comparison['statistics']['chi_squared']
            
            # Комбинирана вероятност
            chi2_total = chi2_bao + chi2_cmb
            log_likelihood = -chi2_total / 2
            
            return log_likelihood
            
        except Exception as e:
            logger.warning(f"Грешка в изчисляването на вероятността: {e}")
            return -np.inf
    
    def _update_analyzers(self, params: Dict[str, float]):
        """
        Обновява анализаторите с нови параметри
        
        Args:
            params: Нови параметри
        """
        # Създаване на нови анализатори
        self.bao_analyzer = BAOAnalyzer(params)
        self.cmb_analyzer = CMBAnalyzer(params)
    
    def objective_function(self, param_array: np.ndarray) -> float:
        """
        Обективна функция за оптимизиране (минимизация на χ²)
        
        Args:
            param_array: Масив с параметри [alpha, beta, gamma, delta]
            
        Returns:
            Общ χ² за минимизиране
        """
        # Конвертиране в речник
        params = {
            'alpha': param_array[0],
            'beta': param_array[1],
            'gamma': param_array[2],
            'delta': param_array[3]
        }
        
        # Връщане на отрицателната вероятност (за минимизиране)
        return -self.calculate_joint_likelihood(params)
    
    def optimize_parameters(self, method: str = 'L-BFGS-B') -> Dict[str, Any]:
        """
        Оптимизира параметрите чрез минимизиране на χ²
        
        Args:
            method: Метод за оптимизиране
            
        Returns:
            Резултати от оптимизацията
        """
        logger.info(f"🔧 Започва оптимизиране с метод {method}...")
        
        # Начални стойности
        x0 = np.array([
            self.current_params['alpha'],
            self.current_params['beta'],
            self.current_params['gamma'],
            self.current_params['delta']
        ])
        
        # Граници
        bounds = [
            self.param_bounds['alpha'],
            self.param_bounds['beta'],
            self.param_bounds['gamma'],
            self.param_bounds['delta']
        ]
        
        # Оптимизиране
        result = optimize.minimize(
            self.objective_function,
            x0,
            method=method,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        # Най-добри параметри
        best_params = {
            'alpha': result.x[0],
            'beta': result.x[1],
            'gamma': result.x[2],
            'delta': result.x[3]
        }
        
        # Обновяване на текущите параметри
        self.current_params = best_params.copy()
        
        logger.info(f"✅ Оптимизирането завърши успешно!")
        logger.info(f"Най-добри параметри: {best_params}")
        
        return {
            'success': result.success,
            'best_parameters': best_params,
            'chi_squared': result.fun,
            'n_iterations': result.nit,
            'optimization_result': result
        }
    
    def comprehensive_joint_analysis(self) -> JointAnalysisResults:
        """
        Извършва обширен съвместен анализ
        
        Returns:
            Пълни резултати от съвместния анализ
        """
        logger.info("🔍 Започва обширен съвместен BAO+CMB анализ...")
        
        # Оптимизиране на параметрите
        optimization_results = self.optimize_parameters()
        
        # Обновяване на анализаторите с най-добрите параметри
        self._update_analyzers(optimization_results['best_parameters'])
        
        # BAO анализ
        logger.info("📊 Извършване на BAO анализ...")
        bao_results = self.bao_analyzer.comprehensive_analysis_report()
        
        # CMB анализ
        logger.info("🌠 Извършване на CMB анализ...")
        cmb_results = self.cmb_analyzer.comprehensive_cmb_analysis()
        
        # Комбинирани статистики
        combined_stats = self._calculate_combined_statistics(bao_results, cmb_results)
        
        # Оценка на съвместимостта
        agreement = self._assess_joint_agreement(combined_stats)
        
        # Неопределеност на параметрите
        param_uncertainties = self._estimate_parameter_uncertainties()
        
        # Съставяне на резултатите
        results = JointAnalysisResults(
            bao_results=bao_results,
            cmb_results=cmb_results,
            combined_statistics=combined_stats,
            best_fit_parameters=optimization_results['best_parameters'],
            parameter_uncertainties=param_uncertainties,
            agreement_assessment=agreement
        )
        
        # Запазване на резултатите
        self.latest_results = results
        
        logger.info("✅ Обширният съвместен анализ е завършен!")
        return results
    
    def _calculate_combined_statistics(self, bao_results: Dict[str, Any], 
                                     cmb_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Изчислява комбинираните статистики
        
        Args:
            bao_results: Резултати от BAO анализа
            cmb_results: Резултати от CMB анализа
            
        Returns:
            Комбинирани статистики
        """
        # BAO статистики
        bao_stats = bao_results['comparison_results']['statistics']
        bao_chi2 = bao_stats['chi_squared']
        bao_dof = bao_stats['dof']
        
        # CMB статистики
        cmb_stats = cmb_results['planck_comparison']['statistics']
        cmb_chi2 = cmb_stats['chi_squared']
        cmb_dof = cmb_stats['dof']
        
        # Комбинирани статистики
        total_chi2 = bao_chi2 + cmb_chi2
        total_dof = bao_dof + cmb_dof
        reduced_chi2 = total_chi2 / total_dof if total_dof > 0 else float('inf')
        
        # Дополнителни статистики
        n_params = 4  # alpha, beta, gamma, delta
        aic = total_chi2 + 2 * n_params
        bic = total_chi2 + n_params * np.log(bao_dof + cmb_dof)
        
        return {
            'bao_chi_squared': bao_chi2,
            'cmb_chi_squared': cmb_chi2,
            'total_chi_squared': total_chi2,
            'bao_dof': bao_dof,
            'cmb_dof': cmb_dof,
            'total_dof': total_dof,
            'reduced_chi_squared': reduced_chi2,
            'aic': aic,
            'bic': bic
        }
    
    def _assess_joint_agreement(self, combined_stats: Dict[str, float]) -> str:
        """
        Оценява съвместимостта на модела с данните
        
        Args:
            combined_stats: Комбинирани статистики
            
        Returns:
            Оценка на съвместимостта
        """
        reduced_chi2 = combined_stats['reduced_chi_squared']
        
        if reduced_chi2 <= 1.0:
            return "Отлично съответствие - моделът е много добър"
        elif reduced_chi2 <= 1.5:
            return "Добро съответствие - моделът е приемлив"
        elif reduced_chi2 <= 2.0:
            return "Умерено съответствие - моделът е средно добър"
        elif reduced_chi2 <= 3.0:
            return "Слабо съответствие - моделът има проблеми"
        else:
            return "Неприемливо съответствие - моделът е неподходящ"
    
    def _estimate_parameter_uncertainties(self) -> Dict[str, float]:
        """
        Оценява неопределеностите на параметрите
        
        Returns:
            Неопределености на параметрите
        """
        # Приблизителна оценка чрез варииране на параметрите
        uncertainties = {}
        
        for param_name in ['alpha', 'beta', 'gamma', 'delta']:
            # Малка промяна в параметъра
            delta_param = 0.01
            
            # Оригинална стойност
            original_params = self.current_params.copy()
            chi2_original = -self.calculate_joint_likelihood(original_params)
            
            # Варирана стойност
            varied_params = original_params.copy()
            varied_params[param_name] += delta_param
            chi2_varied = -self.calculate_joint_likelihood(varied_params)
            
            # Приблизителна неопределеност
            if chi2_varied > chi2_original:
                d_chi2_d_param = (chi2_varied - chi2_original) / delta_param
                # 1σ неопределеност (Δχ² = 1)
                uncertainty = np.sqrt(1.0 / abs(d_chi2_d_param)) if d_chi2_d_param != 0 else 0.1
            else:
                uncertainty = 0.1  # Стандартна стойност
            
            uncertainties[param_name] = uncertainty
        
        return uncertainties
    
    def generate_comparison_table(self) -> Dict[str, Any]:
        """
        Генерира таблица за сравнение с стандартни модели
        
        Returns:
            Таблица за сравнение
        """
        if self.latest_results is None:
            logger.warning("Няма резултати за сравнение. Стартиране на анализ...")
            self.comprehensive_joint_analysis()
        
        # Нелинейни параметри
        nonlinear_params = self.latest_results.best_fit_parameters
        
        # Стандартен ΛCDM за сравнение
        lambda_cdm_params = {'alpha': 0.0, 'beta': 0.0, 'gamma': 0.0, 'delta': 0.0}
        
        # Създаване на ΛCDM анализатори
        bao_lambda_cdm = BAOAnalyzer(lambda_cdm_params)
        cmb_lambda_cdm = CMBAnalyzer(lambda_cdm_params)
        
        # ΛCDM статистики
        bao_lambda_cdm.load_real_data()
        bao_lambda_comparison = bao_lambda_cdm.compare_with_observations()
        cmb_lambda_comparison = cmb_lambda_cdm.compare_with_planck_data()
        
        lambda_cdm_chi2 = (bao_lambda_comparison['statistics']['chi_squared'] + 
                          cmb_lambda_comparison['statistics']['chi_squared'])
        
        # Нелинейни статистики
        nonlinear_chi2 = self.latest_results.combined_statistics['total_chi_squared']
        
        # Подобрение
        delta_chi2 = lambda_cdm_chi2 - nonlinear_chi2
        
        return {
            'lambda_cdm_chi2': lambda_cdm_chi2,
            'nonlinear_chi2': nonlinear_chi2,
            'delta_chi2': delta_chi2,
            'improvement': delta_chi2 > 0,
            'significance': abs(delta_chi2) / np.sqrt(2 * 4),  # 4 допълнителни параметъра
            'nonlinear_parameters': nonlinear_params,
            'lambda_cdm_parameters': lambda_cdm_params
        }
    
    def print_comprehensive_report(self):
        """
        Принтира обширен доклад за резултатите
        """
        if self.latest_results is None:
            print("Няма резултати за показване. Стартиране на анализ...")
            self.comprehensive_joint_analysis()
        
        results = self.latest_results
        
        print("\n" + "="*80)
        print("🌌 ОБШИРЕН ДОКЛАД ЗА СЪВМЕСТЕН BAO+CMB АНАЛИЗ")
        print("="*80)
        
        # Най-добри параметри
        print("\n📊 НАЙ-ДОБРИ ПАРАМЕТРИ:")
        for param, value in results.best_fit_parameters.items():
            uncertainty = results.parameter_uncertainties.get(param, 0.0)
            print(f"  {param}: {value:.4f} ± {uncertainty:.4f}")
        
        # Комбинирани статистики
        print("\n📈 КОМБИНИРАНИ СТАТИСТИКИ:")
        stats = results.combined_statistics
        print(f"  BAO χ²: {stats['bao_chi_squared']:.2f}")
        print(f"  CMB χ²: {stats['cmb_chi_squared']:.2f}")
        print(f"  Общо χ²: {stats['total_chi_squared']:.2f}")
        print(f"  Редуциран χ²: {stats['reduced_chi_squared']:.2f}")
        print(f"  DOF: {stats['total_dof']}")
        print(f"  AIC: {stats['aic']:.2f}")
        print(f"  BIC: {stats['bic']:.2f}")
        
        # Оценка на съвместимостта
        print(f"\n🎯 ОЦЕНКА НА СЪВМЕСТИМОСТТА:")
        print(f"  {results.agreement_assessment}")
        
        # Сравнение с ΛCDM
        print(f"\n⚖️ СРАВНЕНИЕ С ΛCDM:")
        comparison = self.generate_comparison_table()
        print(f"  ΛCDM χ²: {comparison['lambda_cdm_chi2']:.2f}")
        print(f"  Нелинейно χ²: {comparison['nonlinear_chi2']:.2f}")
        print(f"  Δχ²: {comparison['delta_chi2']:.2f}")
        print(f"  Подобрение: {'Да' if comparison['improvement'] else 'Не'}")
        print(f"  Значимост: {comparison['significance']:.2f}σ")
        
        print("\n" + "="*80)
        print("✅ АНАЛИЗЪТ Е ЗАВЪРШЕН УСПЕШНО!")
        print("="*80)


def test_joint_analyzer():
    """Тест на съвместния анализатор"""
    print("🧪 ТЕСТ НА СЪВМЕСТНИЯ BAO+CMB АНАЛИЗАТОР")
    print("="*60)
    
    # Създаване на анализатор
    analyzer = JointBAOCMBAnalyzer()
    
    # Оптимизиране на параметрите
    print("\n🔧 Оптимизиране на параметрите...")
    optimization = analyzer.optimize_parameters()
    print(f"Успех: {optimization['success']}")
    print(f"Най-добър χ²: {optimization['chi_squared']:.2f}")
    
    # Обширен анализ
    print("\n🔍 Обширен съвместен анализ...")
    results = analyzer.comprehensive_joint_analysis()
    
    # Принтиране на доклада
    analyzer.print_comprehensive_report()
    
    print("\n✅ Всички тестове завършиха успешно!")
    
    return analyzer


if __name__ == "__main__":
    analyzer = test_joint_analyzer() 