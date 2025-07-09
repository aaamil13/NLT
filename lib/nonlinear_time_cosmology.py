"""
Нелинейно Време Космология (NonLinear Time Cosmology)
Библиотека за моделиране на абсолютни и релативни координатни системи

Автор: Основано на концепцията за абсолютни координатни системи
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# Константи
PLANCK_TIME = 5.391e-44  # секунди
HUBBLE_TIME = 13.8e9 * 365.25 * 24 * 3600  # секунди
CRITICAL_DENSITY = 9.47e-27  # kg/m³

@dataclass
class CosmologicalParameters:
    """Космологични параметри за модела"""
    # Параметри за времето
    initial_density: float = 1e30  # kg/m³
    current_density: float = 2.775e-27  # kg/m³
    
    # Параметри за разширението
    linear_expansion_rate: float = 1.0  # в АКС
    time_scaling_exponent: float = 0.5  # експонент за времевото мащабиране
    
    # Общи параметри
    universe_age_abs: float = 13.8e9  # години в абсолютно време
    universe_age_rel: float = 13.8e9  # години в релативно време

class AbsoluteCoordinateSystem:
    """
    Абсолютна Координатна Система (АКС)
    Фиксирана във времето система с точни геометрични координати
    """
    
    def __init__(self, time_abs: float, params: CosmologicalParameters):
        """
        Инициализира АКС за определен момент в абсолютното време
        
        Args:
            time_abs: Абсолютно време в години
            params: Космологични параметри
        """
        self.time_abs = time_abs
        self.params = params
        self.scale_factor = self._calculate_scale_factor()
        self.density = self._calculate_density()
        self.time_rate = self._calculate_time_rate()
        
    def _calculate_scale_factor(self) -> float:
        """Изчислява мащабен фактор за АКС (линейно разширение)"""
        # В АКС разширението е линейно: a(t) = k * t
        k = self.params.linear_expansion_rate
        return k * self.time_abs
    
    def _calculate_density(self) -> float:
        """Изчислява плътност в АКС"""
        # Плътността намалява обратно пропорционално на обема
        # ρ(t) = ρ₀ * (a₀/a(t))³
        a0 = self.params.linear_expansion_rate * 1e9  # първа година
        return self.params.initial_density * (a0 / self.scale_factor)**3
    
    def _calculate_time_rate(self) -> float:
        """Изчислява темпа на време в АКС"""
        # τ(t) = (ρ(t)/ρ₀)^(-k) където k ∈ [1/3, 1]
        k = self.params.time_scaling_exponent
        return (self.density / self.params.current_density)**(-k)
    
    def get_coordinates(self, object_id: str) -> np.ndarray:
        """Връща абсолютни координати за обект в тази АКС"""
        # Симулирани координати за демонстрация
        np.random.seed(hash(object_id) % 2**32)
        return np.random.uniform(-1, 1, 3) * self.scale_factor
    
    def __str__(self) -> str:
        return f"АКС(t={self.time_abs:.2e} години, a={self.scale_factor:.2e}, ρ={self.density:.2e})"

class RelativeCoordinateSystem:
    """
    Релативна Координатна Система (РКС)
    Отместена във времето система спрямо АКС
    """
    
    def __init__(self, observation_time: float, params: CosmologicalParameters):
        """
        Инициализира РКС за наблюдение от определен момент
        
        Args:
            observation_time: Време на наблюдение в години
            params: Космологични параметри
        """
        self.observation_time = observation_time
        self.params = params
        self.scale_factor = self._calculate_relative_scale_factor()
        
    def _calculate_relative_scale_factor(self) -> float:
        """Изчислява мащабен фактор за РКС (кубично разширение)"""
        # В РКС разширението изглежда кубично поради времевото изкривяване
        # a_rel(t) = k * t³ (или друга степен > 1)
        k = self.params.linear_expansion_rate / (1e9**2)  # нормализация
        return k * self.observation_time**3
    
    def transform_from_abs(self, abs_coords: np.ndarray, abs_time: float) -> np.ndarray:
        """
        Трансформира координати от АКС към РКС
        
        Args:
            abs_coords: Абсолютни координати
            abs_time: Абсолютно време
            
        Returns:
            Трансформирани координати в РКС
        """
        # Времево изместване създава деформация на координатите
        time_displacement = self.observation_time - abs_time
        
        if time_displacement <= 0:
            return abs_coords
            
        # Времевото изместване създава координатно разтягане
        expansion_factor = self._calculate_expansion_factor(time_displacement)
        
        # Редшифт ефект
        z = self._calculate_redshift(time_displacement)
        
        return abs_coords * expansion_factor * (1 + z)
    
    def _calculate_expansion_factor(self, time_displacement: float) -> float:
        """Изчислява фактор на разширение поради времевото изместване"""
        # Експоненциален растеж на разширението с времето
        return np.exp(time_displacement / self.params.universe_age_abs)
    
    def _calculate_redshift(self, time_displacement: float) -> float:
        """Изчислява редшифт z поради времевото изместване"""
        # z = (λ_obs - λ_emit) / λ_emit
        # Приближение: z ≈ time_displacement / universe_age
        return time_displacement / self.params.universe_age_abs
    
    def __str__(self) -> str:
        return f"РКС(t_obs={self.observation_time:.2e} години, a={self.scale_factor:.2e})"

class ExpansionCalculator:
    """
    Калкулатор за коефициенти на разширение между АКС и РКС
    """
    
    def __init__(self, params: CosmologicalParameters):
        self.params = params
    
    def calculate_abs_expansion_coefficient(self, time1: float, time2: float) -> float:
        """
        Изчислява коефициент на разширение между две АКС
        
        Args:
            time1: Първо време в години
            time2: Второ време в години
            
        Returns:
            Коефициент на разширение R = a(t2)/a(t1)
        """
        acs1 = AbsoluteCoordinateSystem(time1, self.params)
        acs2 = AbsoluteCoordinateSystem(time2, self.params)
        
        return acs2.scale_factor / acs1.scale_factor
    
    def calculate_rel_expansion_coefficient(self, time1: float, time2: float) -> float:
        """
        Изчислява коефициент на разширение между две РКС
        
        Args:
            time1: Първо време в години
            time2: Второ време в години
            
        Returns:
            Коефициент на разширение R = a(t2)/a(t1)
        """
        rcs1 = RelativeCoordinateSystem(time1, self.params)
        rcs2 = RelativeCoordinateSystem(time2, self.params)
        
        return rcs2.scale_factor / rcs1.scale_factor
    
    def check_linearity(self, time_points: List[float], system_type: str = "abs") -> Dict:
        """
        Проверява дали разширението е линейно за дадени времеви точки
        
        Args:
            time_points: Списък от времеви точки в години
            system_type: "abs" за АКС или "rel" за РКС
            
        Returns:
            Речник с резултати от проверката
        """
        coefficients = []
        
        for i in range(len(time_points) - 1):
            if system_type == "abs":
                coeff = self.calculate_abs_expansion_coefficient(time_points[i], time_points[i+1])
            else:
                coeff = self.calculate_rel_expansion_coefficient(time_points[i], time_points[i+1])
            coefficients.append(coeff)
        
        # Проверка за константност на коефициентите
        mean_coeff = np.mean(coefficients)
        std_coeff = np.std(coefficients)
        linearity_measure = std_coeff / mean_coeff if mean_coeff != 0 else float('inf')
        
        return {
            'coefficients': coefficients,
            'mean_coefficient': mean_coeff,
            'std_coefficient': std_coeff,
            'linearity_measure': linearity_measure,
            'is_linear': linearity_measure < 0.1  # произволен праг
        }
    
    def compare_expansion_types(self, time_points: List[float]) -> Dict:
        """
        Сравнява линейното разширение в АКС с кубичното в РКС
        
        Args:
            time_points: Списък от времеви точки в години
            
        Returns:
            Речник със сравнителни резултати
        """
        abs_results = self.check_linearity(time_points, "abs")
        rel_results = self.check_linearity(time_points, "rel")
        
        return {
            'abs_system': abs_results,
            'rel_system': rel_results,
            'linearity_difference': abs_results['linearity_measure'] - rel_results['linearity_measure']
        }

class CosmologyVisualizer:
    """
    Визуализатор за космологичните модели
    """
    
    def __init__(self, params: CosmologicalParameters):
        self.params = params
        self.calculator = ExpansionCalculator(params)
    
    def plot_expansion_comparison(self, time_range: Tuple[float, float], num_points: int = 100):
        """
        Графика за сравнение на разширението в АКС и РКС
        
        Args:
            time_range: Обхват на времето (начало, край) в години
            num_points: Брой точки за изчисление
        """
        times = np.linspace(time_range[0], time_range[1], num_points)
        
        # Изчисляване на мащабни фактори
        abs_scale_factors = []
        rel_scale_factors = []
        
        for t in times:
            abs_acs = AbsoluteCoordinateSystem(t, self.params)
            rel_rcs = RelativeCoordinateSystem(t, self.params)
            
            abs_scale_factors.append(abs_acs.scale_factor)
            rel_scale_factors.append(rel_rcs.scale_factor)
        
        # Създаване на графика
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(times, abs_scale_factors, 'b-', label='АКС (линейно)')
        plt.xlabel('Време (години)')
        plt.ylabel('Мащабен фактор')
        plt.title('Разширение в АКС')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(times, rel_scale_factors, 'r-', label='РКС (кубично)')
        plt.xlabel('Време (години)')
        plt.ylabel('Мащабен фактор')
        plt.title('Разширение в РКС')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(times, abs_scale_factors, 'b-', label='АКС')
        plt.plot(times, rel_scale_factors, 'r-', label='РКС')
        plt.xlabel('Време (години)')
        plt.ylabel('Мащабен фактор')
        plt.title('Сравнение АКС vs РКС')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        ratio = np.array(rel_scale_factors) / np.array(abs_scale_factors)
        plt.plot(times, ratio, 'g-', label='РКС/АКС')
        plt.xlabel('Време (години)')
        plt.ylabel('Отношение')
        plt.title('Отношение РКС/АКС')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_expansion_coefficients(self, time_points: List[float]):
        """
        Графика на коефициентите на разширение
        
        Args:
            time_points: Списък от времеви точки в години
        """
        comparison = self.calculator.compare_expansion_types(time_points)
        
        abs_coeffs = comparison['abs_system']['coefficients']
        rel_coeffs = comparison['rel_system']['coefficients']
        
        intervals = [f"{time_points[i]:.1e}-{time_points[i+1]:.1e}" for i in range(len(time_points)-1)]
        
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(intervals))
        width = 0.35
        
        plt.bar(x - width/2, abs_coeffs, width, label='АКС', color='blue', alpha=0.7)
        plt.bar(x + width/2, rel_coeffs, width, label='РКС', color='red', alpha=0.7)
        
        plt.xlabel('Времеви интервали (години)')
        plt.ylabel('Коефициент на разширение')
        plt.title('Коефициенти на разширение: АКС vs РКС')
        plt.xticks(x, intervals, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def create_demo_scenario():
    """Демонстрационен сценарий за използване на библиотеката"""
    
    # Създаване на параметри
    params = CosmologicalParameters(
        initial_density=1e30,
        current_density=2.775e-27,
        linear_expansion_rate=1.0,
        time_scaling_exponent=0.5,
        universe_age_abs=13.8e9,
        universe_age_rel=13.8e9
    )
    
    # Времеви точки за анализ (в години)
    time_points = [1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9, 10e9]
    
    # Създаване на калкулатор
    calculator = ExpansionCalculator(params)
    
    # Анализ на линейността
    print("=== АНАЛИЗ НА ЛИНЕЙНОСТТА ===")
    comparison = calculator.compare_expansion_types(time_points)
    
    print(f"АКС - Среден коефициент: {comparison['abs_system']['mean_coefficient']:.4f}")
    print(f"АКС - Стандартно отклонение: {comparison['abs_system']['std_coefficient']:.4f}")
    print(f"АКС - Мярка за линейност: {comparison['abs_system']['linearity_measure']:.4f}")
    print(f"АКС - Линейно: {comparison['abs_system']['is_linear']}")
    
    print(f"\nРКС - Среден коефициент: {comparison['rel_system']['mean_coefficient']:.4f}")
    print(f"РКС - Стандартно отклонение: {comparison['rel_system']['std_coefficient']:.4f}")
    print(f"РКС - Мярка за линейност: {comparison['rel_system']['linearity_measure']:.4f}")
    print(f"РКС - Линейно: {comparison['rel_system']['is_linear']}")
    
    # Създаване на няколко АКС
    print("\n=== СЪЗДАВАНЕ НА АКС ===")
    for t in [1e9, 5e9, 10e9]:
        acs = AbsoluteCoordinateSystem(t, params)
        print(acs)
    
    # Създаване на РКС
    print("\n=== СЪЗДАВАНЕ НА РКС ===")
    for t in [1e9, 5e9, 10e9]:
        rcs = RelativeCoordinateSystem(t, params)
        print(rcs)
    
    # Визуализация
    print("\n=== ВИЗУАЛИЗАЦИЯ ===")
    visualizer = CosmologyVisualizer(params)
    
    # Създаване на графики
    visualizer.plot_expansion_comparison((1e9, 10e9))
    visualizer.plot_expansion_coefficients(time_points)
    
    return params, calculator, visualizer

if __name__ == "__main__":
    # Стартиране на демонстрацията
    params, calculator, visualizer = create_demo_scenario() 