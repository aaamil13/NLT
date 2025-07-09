"""
Калибриране на скоростта на разширение спрямо червеното отместване
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from .nonlinear_time_cosmology import *

@dataclass
class RedshiftData:
    """Данни за червено отместване"""
    redshift: float  # z стойност
    distance: float  # разстояние в Mpc
    age: float       # възраст в години
    
@dataclass
class ObservationalData:
    """Наблюдателни данни за калибриране"""
    redshift_points: List[RedshiftData]
    current_age: float = 13.8e9  # години
    hubble_constant: float = 70.0  # km/s/Mpc

class LinearTimeStepGenerator:
    """
    Генератор на линейни времеви стъпки за АКС
    """
    
    def __init__(self, start_time: float, end_time: float, step_size: float):
        """
        Инициализира генератор на времеви стъпки
        
        Args:
            start_time: Начално време в години
            end_time: Крайно време в години
            step_size: Размер на стъпката в години
        """
        self.start_time = start_time
        self.end_time = end_time
        self.step_size = step_size
        self.time_steps = self._generate_time_steps()
    
    def _generate_time_steps(self) -> List[float]:
        """Генерира равномерни времеви стъпки"""
        return list(np.arange(self.start_time, self.end_time + self.step_size, self.step_size))
    
    def get_time_steps(self) -> List[float]:
        """Връща генерираните времеви стъпки"""
        return self.time_steps
    
    def get_acs_sequence(self, params: CosmologicalParameters) -> List[AbsoluteCoordinateSystem]:
        """
        Създава последователност от АКС за времевите стъпки
        
        Args:
            params: Космологични параметри
            
        Returns:
            Списък от АКС системи
        """
        return [AbsoluteCoordinateSystem(t, params) for t in self.time_steps]

class RedshiftCalculator:
    """
    Калкулатор за червено отместване в модела
    """
    
    def __init__(self, params: CosmologicalParameters):
        self.params = params
    
    def calculate_redshift_from_age(self, emission_age: float, observation_age: float) -> float:
        """
        Изчислява червено отместване от възраст на емисия и наблюдение
        
        Args:
            emission_age: Възраст при емисия в години
            observation_age: Възраст при наблюдение в години
            
        Returns:
            Червено отместване z
        """
        # Създаваме АКС за двете времена
        acs_emission = AbsoluteCoordinateSystem(emission_age, self.params)
        acs_observation = AbsoluteCoordinateSystem(observation_age, self.params)
        
        # Червеното отместване е свързано с отношението на мащабните фактори
        # z = (a_obs / a_emit) - 1
        z = (acs_observation.scale_factor / acs_emission.scale_factor) - 1
        
        return z
    
    def calculate_distance_from_redshift(self, z: float, observation_age: float) -> float:
        """
        Изчислява разстояние от червено отместване
        
        Args:
            z: Червено отместване
            observation_age: Възраст при наблюдение в години
            
        Returns:
            Разстояние в Mpc
        """
        # Опростен модел: d ≈ c * z / H0
        # където c е скоростта на светлината, H0 е константата на Хъбъл
        c = 299792.458  # km/s
        H0 = 70.0  # km/s/Mpc
        
        return c * z / H0
    
    def calculate_age_from_redshift(self, z: float, observation_age: float) -> float:
        """
        Изчислява възраст при емисия от червено отместване
        
        Args:
            z: Червено отместване
            observation_age: Възраст при наблюдение в години
            
        Returns:
            Възраст при емисия в години
        """
        # Обратно изчисление: a_emit = a_obs / (1 + z)
        acs_observation = AbsoluteCoordinateSystem(observation_age, self.params)
        a_emit = acs_observation.scale_factor / (1 + z)
        
        # Намираме времето за това мащабиране
        # a = k * t => t = a / k
        emission_age = a_emit / self.params.linear_expansion_rate
        
        return emission_age

class ExpansionRateCalibrator:
    """
    Калибратор за скоростта на разширение спрямо наблюдателни данни
    """
    
    def __init__(self, observational_data: ObservationalData):
        self.observational_data = observational_data
    
    def create_standard_redshift_data(self) -> List[RedshiftData]:
        """
        Създава стандартни данни за червено отместване основани на наблюдения
        """
        # Стандартни z стойности и съответстващи възрасти
        standard_data = [
            (0.1, 12.5e9),   # z=0.1, възраст 12.5 млрд години
            (0.5, 10.0e9),   # z=0.5, възраст 10.0 млрд години  
            (1.0, 7.5e9),    # z=1.0, възраст 7.5 млрд години
            (2.0, 5.0e9),    # z=2.0, възраст 5.0 млрд години
            (3.0, 3.5e9),    # z=3.0, възраст 3.5 млрд години
            (5.0, 2.0e9),    # z=5.0, възраст 2.0 млрд години
            (7.0, 1.2e9),    # z=7.0, възраст 1.2 млрд години
            (10.0, 0.8e9),   # z=10.0, възраст 0.8 млрд години
        ]
        
        redshift_data = []
        for z, age in standard_data:
            # Опростено изчисление на разстоянието
            distance = 299792.458 * z / 70.0  # Mpc
            redshift_data.append(RedshiftData(z, distance, age))
        
        return redshift_data
    
    def objective_function(self, expansion_rate: float, redshift_data: List[RedshiftData]) -> float:
        """
        Целева функция за оптимизация на скоростта на разширение
        
        Args:
            expansion_rate: Скорост на разширение k
            redshift_data: Наблюдателни данни
            
        Returns:
            Стойност на грешката
        """
        # Създаваме параметри с тестваната скорост на разширение
        params = CosmologicalParameters(linear_expansion_rate=expansion_rate)
        calculator = RedshiftCalculator(params)
        
        total_error = 0.0
        observation_age = self.observational_data.current_age
        
        for data_point in redshift_data:
            # Изчисляваме предсказаното червено отместване
            predicted_z = calculator.calculate_redshift_from_age(
                data_point.age, observation_age
            )
            
            # Изчисляваме грешката
            error = (predicted_z - data_point.redshift)**2
            total_error += error
        
        return total_error
    
    def calibrate_expansion_rate(self, initial_guess: float = 1.0) -> Dict:
        """
        Калибрира скоростта на разширение
        
        Args:
            initial_guess: Начална стойност за скоростта
            
        Returns:
            Резултати от калибрирането
        """
        # Получаваме стандартни данни
        redshift_data = self.create_standard_redshift_data()
        
        # Дефинираме целева функция
        def objective(expansion_rate):
            return self.objective_function(expansion_rate, redshift_data)
        
        # Оптимизираме
        result = minimize_scalar(
            objective,
            bounds=(0.1, 10.0),
            method='bounded'
        )
        
        optimal_rate = result.x
        final_error = result.fun
        
        # Изчисляваме детайлни резултати
        params = CosmologicalParameters(linear_expansion_rate=optimal_rate)
        calculator = RedshiftCalculator(params)
        
        predictions = []
        for data_point in redshift_data:
            predicted_z = calculator.calculate_redshift_from_age(
                data_point.age, self.observational_data.current_age
            )
            predictions.append({
                'observed_z': data_point.redshift,
                'predicted_z': predicted_z,
                'error': abs(predicted_z - data_point.redshift),
                'age': data_point.age
            })
        
        return {
            'optimal_expansion_rate': optimal_rate,
            'final_error': final_error,
            'predictions': predictions,
            'success': result.success,
            'optimization_result': result
        }

class RedshiftComparisonVisualizer:
    """
    Визуализатор за сравнение на модела с наблюдения
    """
    
    def __init__(self, calibrator: ExpansionRateCalibrator):
        self.calibrator = calibrator
    
    def plot_redshift_comparison(self, calibration_results: Dict):
        """
        Създава графики за сравнение на модела с наблюдения
        
        Args:
            calibration_results: Резултати от калибрирането
        """
        predictions = calibration_results['predictions']
        
        observed_z = [p['observed_z'] for p in predictions]
        predicted_z = [p['predicted_z'] for p in predictions]
        ages = [p['age'] / 1e9 for p in predictions]  # в млрд години
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Наблюдавано vs Предсказано червено отместване
        axes[0, 0].scatter(observed_z, predicted_z, color='blue', s=100, alpha=0.7)
        axes[0, 0].plot([0, max(observed_z)], [0, max(observed_z)], 'r--', label='Идеално съвпадение')
        axes[0, 0].set_xlabel('Наблюдавано z')
        axes[0, 0].set_ylabel('Предсказано z')
        axes[0, 0].set_title('Наблюдавано vs Предсказано червено отместване')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Червено отместване vs Възраст
        axes[0, 1].scatter(ages, observed_z, color='red', label='Наблюдавано', s=100)
        axes[0, 1].scatter(ages, predicted_z, color='blue', label='Предсказано', s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Възраст (млрд години)')
        axes[0, 1].set_ylabel('Червено отместване z')
        axes[0, 1].set_title('Червено отместване vs Възраст')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Относителна грешка
        relative_errors = [(p['error'] / p['observed_z']) * 100 for p in predictions]
        axes[1, 0].bar(range(len(ages)), relative_errors, color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Точка от данните')
        axes[1, 0].set_ylabel('Относителна грешка (%)')
        axes[1, 0].set_title('Относителна грешка за всяка точка')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Разлики между наблюдавано и предсказано
        differences = [p['predicted_z'] - p['observed_z'] for p in predictions]
        axes[1, 1].bar(range(len(ages)), differences, color='green', alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_xlabel('Точка от данните')
        axes[1, 1].set_ylabel('Разлика (предсказано - наблюдавано)')
        axes[1, 1].set_title('Разлики между предсказания и наблюдения')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Добавяме информация за оптималната скорост
        optimal_rate = calibration_results['optimal_expansion_rate']
        final_error = calibration_results['final_error']
        
        fig.suptitle(f'Калибриране на скоростта на разширение\n'
                    f'Оптимална скорост: {optimal_rate:.6f}, Крайна грешка: {final_error:.6f}',
                    fontsize=16)
        
        plt.tight_layout()
        plt.show()
    
    def plot_expansion_rate_sensitivity(self, rate_range: Tuple[float, float], num_points: int = 50):
        """
        Създава графика за чувствителност на грешката към скоростта на разширение
        
        Args:
            rate_range: Обхват на скоростите за тестване
            num_points: Брой точки за тестване
        """
        rates = np.linspace(rate_range[0], rate_range[1], num_points)
        errors = []
        
        redshift_data = self.calibrator.create_standard_redshift_data()
        
        for rate in rates:
            error = self.calibrator.objective_function(rate, redshift_data)
            errors.append(error)
        
        plt.figure(figsize=(10, 6))
        plt.plot(rates, errors, 'b-', linewidth=2)
        plt.xlabel('Скорост на разширение k')
        plt.ylabel('Сума от квадратите на грешките')
        plt.title('Чувствителност на грешката към скоростта на разширение')
        plt.grid(True, alpha=0.3)
        
        # Намираме и маркираме минимума
        min_idx = np.argmin(errors)
        plt.plot(rates[min_idx], errors[min_idx], 'ro', markersize=10, label=f'Минимум при k={rates[min_idx]:.4f}')
        plt.legend()
        
        plt.show()

def create_linear_time_step_analysis(step_size: float = 1e9, start_time: float = 1e9, end_time: float = 13.8e9):
    """
    Създава анализ с линейни времеви стъпки
    
    Args:
        step_size: Размер на стъпката в години
        start_time: Начално време в години
        end_time: Крайно време в години
    """
    print(f"=== АНАЛИЗ С ЛИНЕЙНИ ВРЕМЕВИ СТЪПКИ ===")
    print(f"Стъпка: {step_size/1e9:.1f} млрд години")
    print(f"Обхват: {start_time/1e9:.1f} - {end_time/1e9:.1f} млрд години")
    
    # Създаваме данни за наблюдения
    obs_data = ObservationalData([], current_age=13.8e9)
    
    # Създаваме калибратор
    calibrator = ExpansionRateCalibrator(obs_data)
    
    # Калибриране на скоростта на разширение
    print("\nКалибриране на скоростта на разширение...")
    calibration_results = calibrator.calibrate_expansion_rate()
    
    optimal_rate = calibration_results['optimal_expansion_rate']
    print(f"Оптимална скорост на разширение: {optimal_rate:.6f}")
    print(f"Крайна грешка: {calibration_results['final_error']:.6f}")
    
    # Създаваме оптимизирани параметри
    params = CosmologicalParameters(linear_expansion_rate=optimal_rate)
    
    # Генерираме линейни времеви стъпки
    step_generator = LinearTimeStepGenerator(start_time, end_time, step_size)
    time_steps = step_generator.get_time_steps()
    acs_sequence = step_generator.get_acs_sequence(params)
    
    print(f"\nГенерирани {len(time_steps)} времеви стъпки:")
    for i, (t, acs) in enumerate(zip(time_steps, acs_sequence)):
        print(f"  Стъпка {i+1}: {t/1e9:.1f} млрд години, a={acs.scale_factor:.2e}")
    
    # Изчисляваме коефициенти на разширение между стъпките
    print(f"\nКоефициенти на разширение между стъпките:")
    calculator = ExpansionCalculator(params)
    
    for i in range(len(time_steps) - 1):
        coeff = calculator.calculate_abs_expansion_coefficient(time_steps[i], time_steps[i+1])
        print(f"  {time_steps[i]/1e9:.1f} -> {time_steps[i+1]/1e9:.1f} млрд години: {coeff:.6f}")
    
    # Визуализация
    visualizer = RedshiftComparisonVisualizer(calibrator)
    visualizer.plot_redshift_comparison(calibration_results)
    visualizer.plot_expansion_rate_sensitivity((0.1, 3.0))
    
    return {
        'calibration_results': calibration_results,
        'time_steps': time_steps,
        'acs_sequence': acs_sequence,
        'optimal_params': params,
        'step_generator': step_generator,
        'visualizer': visualizer
    }

if __name__ == "__main__":
    # Стартираме анализа с линейни времеви стъпки
    results = create_linear_time_step_analysis(
        step_size=1e9,      # 1 млрд години
        start_time=1e9,     # започваме от 1 млрд години
        end_time=13.8e9     # до текущата възраст
    ) 