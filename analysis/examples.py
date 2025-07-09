"""
Примери за използване на библиотеката за нелинейно време космология
"""

from nonlinear_time_cosmology import *
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def example_1_basic_acs_creation():
    """
    Пример 1: Създаване на основни АКС за различни времеви моменти
    """
    print("=== ПРИМЕР 1: СЪЗДАВАНЕ НА АКС ===")
    
    # Параметри на модела
    params = CosmologicalParameters(
        initial_density=1e30,
        current_density=2.775e-27,
        linear_expansion_rate=1.0,
        time_scaling_exponent=0.5
    )
    
    # Времеви моменти (в години)
    times = [1e9, 3e9, 5e9, 7e9, 10e9, 13.8e9]
    
    print("Създаване на АКС за различни времеви моменти:")
    for t in times:
        acs = AbsoluteCoordinateSystem(t, params)
        print(f"  {acs}")
        print(f"    Темп на време: {acs.time_rate:.6f}")
        print(f"    Плътност: {acs.density:.2e} kg/m³")
    
    return params, times

def example_2_expansion_coefficients():
    """
    Пример 2: Изчисляване на коефициенти на разширение между АКС
    """
    print("\n=== ПРИМЕР 2: КОЕФИЦИЕНТИ НА РАЗШИРЕНИЕ ===")
    
    params = CosmologicalParameters()
    calculator = ExpansionCalculator(params)
    
    # Интервали от 1 милиард години
    time_intervals = [
        (1e9, 2e9),
        (2e9, 3e9),
        (3e9, 4e9),
        (4e9, 5e9),
        (5e9, 6e9),
        (6e9, 7e9),
        (7e9, 8e9),
        (8e9, 9e9),
        (9e9, 10e9)
    ]
    
    print("Коефициенти на разширение за интервали от 1 млрд години:")
    print("Време (години)        АКС коефициент    РКС коефициент")
    print("-" * 55)
    
    for t1, t2 in time_intervals:
        abs_coeff = calculator.calculate_abs_expansion_coefficient(t1, t2)
        rel_coeff = calculator.calculate_rel_expansion_coefficient(t1, t2)
        
        print(f"{t1:.0e} - {t2:.0e}     {abs_coeff:.6f}          {rel_coeff:.6f}")
    
    return calculator

def example_3_linearity_analysis():
    """
    Пример 3: Анализ на линейността на разширението
    """
    print("\n=== ПРИМЕР 3: АНАЛИЗ НА ЛИНЕЙНОСТТА ===")
    
    params = CosmologicalParameters()
    calculator = ExpansionCalculator(params)
    
    # Времеви точки за анализ
    time_points = [1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9, 10e9]
    
    # Анализ на линейността
    abs_analysis = calculator.check_linearity(time_points, "abs")
    rel_analysis = calculator.check_linearity(time_points, "rel")
    
    print("Анализ на линейността в АКС:")
    print(f"  Коефициенти: {abs_analysis['coefficients']}")
    print(f"  Среден коефициент: {abs_analysis['mean_coefficient']:.6f}")
    print(f"  Стандартно отклонение: {abs_analysis['std_coefficient']:.6f}")
    print(f"  Мярка за линейност: {abs_analysis['linearity_measure']:.6f}")
    print(f"  Линейно: {abs_analysis['is_linear']}")
    
    print("\nАнализ на линейността в РКС:")
    print(f"  Коефициенти: {rel_analysis['coefficients']}")
    print(f"  Среден коефициент: {rel_analysis['mean_coefficient']:.6f}")
    print(f"  Стандартно отклонение: {rel_analysis['std_coefficient']:.6f}")
    print(f"  Мярка за линейност: {rel_analysis['linearity_measure']:.6f}")
    print(f"  Линейно: {rel_analysis['is_linear']}")
    
    return abs_analysis, rel_analysis

def example_4_coordinate_transformation():
    """
    Пример 4: Трансформация на координати между АКС и РКС
    """
    print("\n=== ПРИМЕР 4: ТРАНСФОРМАЦИЯ НА КООРДИНАТИ ===")
    
    params = CosmologicalParameters()
    
    # Създаване на АКС при 5 млрд години
    acs_5Gy = AbsoluteCoordinateSystem(5e9, params)
    
    # Създаване на РКС при 10 млрд години
    rcs_10Gy = RelativeCoordinateSystem(10e9, params)
    
    # Координати на някакъв обект в АКС
    object_coords_abs = acs_5Gy.get_coordinates("galaxy_123")
    
    print(f"Координати в АКС (5 млрд години): {object_coords_abs}")
    
    # Трансформация към РКС
    object_coords_rel = rcs_10Gy.transform_from_abs(object_coords_abs, 5e9)
    
    print(f"Координати в РКС (10 млрд години): {object_coords_rel}")
    
    # Изчисляване на разтягането
    expansion_factor = np.linalg.norm(object_coords_rel) / np.linalg.norm(object_coords_abs)
    print(f"Фактор на разтягане: {expansion_factor:.6f}")
    
    return object_coords_abs, object_coords_rel, expansion_factor

def example_5_time_evolution():
    """
    Пример 5: Еволюция на времето и пространството
    """
    print("\n=== ПРИМЕР 5: ЕВОЛЮЦИЯ НА ВРЕМЕТО И ПРОСТРАНСТВОТО ===")
    
    params = CosmologicalParameters()
    
    # Времеви серия
    times = np.linspace(1e9, 13.8e9, 20)
    
    # Данни за съхранение
    abs_scale_factors = []
    rel_scale_factors = []
    densities = []
    time_rates = []
    
    for t in times:
        acs = AbsoluteCoordinateSystem(t, params)
        rcs = RelativeCoordinateSystem(t, params)
        
        abs_scale_factors.append(acs.scale_factor)
        rel_scale_factors.append(rcs.scale_factor)
        densities.append(acs.density)
        time_rates.append(acs.time_rate)
    
    # Анализ на темповете на нарастване
    print("Анализ на темповете на нарастване:")
    
    # Средни темпове на нарастване
    abs_growth_rates = np.diff(abs_scale_factors) / np.diff(times)
    rel_growth_rates = np.diff(rel_scale_factors) / np.diff(times)
    
    print(f"Среден темп на нарастване в АКС: {np.mean(abs_growth_rates):.2e}")
    print(f"Среден темп на нарастване в РКС: {np.mean(rel_growth_rates):.2e}")
    print(f"Отношение РКС/АКС: {np.mean(rel_growth_rates) / np.mean(abs_growth_rates):.6f}")
    
    # Корелация между плътност и темп на време
    correlation = np.corrcoef(densities, time_rates)[0, 1]
    print(f"Корелация между плътност и темп на време: {correlation:.6f}")
    
    return times, abs_scale_factors, rel_scale_factors, densities, time_rates

def example_6_advanced_visualization():
    """
    Пример 6: Разширена визуализация на резултатите
    """
    print("\n=== ПРИМЕР 6: РАЗШИРЕНА ВИЗУАЛИЗАЦИЯ ===")
    
    params = CosmologicalParameters()
    visualizer = CosmologyVisualizer(params)
    
    # Времеви точки за анализ
    time_points = [1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9, 10e9, 11e9, 12e9, 13e9]
    
    # Основни графики
    visualizer.plot_expansion_comparison((1e9, 13.8e9))
    visualizer.plot_expansion_coefficients(time_points)
    
    # Допълнителни графики
    create_advanced_plots(params, time_points)
    
    return visualizer

def create_advanced_plots(params: CosmologicalParameters, time_points: List[float]):
    """
    Създава разширени графики за анализ
    """
    # Данни за изчисление
    times = np.array(time_points)
    abs_scale_factors = []
    rel_scale_factors = []
    densities = []
    time_rates = []
    
    for t in times:
        acs = AbsoluteCoordinateSystem(t, params)
        rcs = RelativeCoordinateSystem(t, params)
        
        abs_scale_factors.append(acs.scale_factor)
        rel_scale_factors.append(rcs.scale_factor)
        densities.append(acs.density)
        time_rates.append(acs.time_rate)
    
    abs_scale_factors = np.array(abs_scale_factors)
    rel_scale_factors = np.array(rel_scale_factors)
    densities = np.array(densities)
    time_rates = np.array(time_rates)
    
    # Създаване на графики
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Плътност във времето
    axes[0, 0].loglog(times, densities, 'b-', marker='o')
    axes[0, 0].set_xlabel('Време (години)')
    axes[0, 0].set_ylabel('Плътност (kg/m³)')
    axes[0, 0].set_title('Плътност във времето')
    axes[0, 0].grid(True)
    
    # 2. Темп на време
    axes[0, 1].semilogx(times, time_rates, 'g-', marker='s')
    axes[0, 1].set_xlabel('Време (години)')
    axes[0, 1].set_ylabel('Темп на време')
    axes[0, 1].set_title('Темп на време в АКС')
    axes[0, 1].grid(True)
    
    # 3. Отношение на мащабните фактори
    ratio = rel_scale_factors / abs_scale_factors
    axes[0, 2].semilogy(times, ratio, 'r-', marker='^')
    axes[0, 2].set_xlabel('Време (години)')
    axes[0, 2].set_ylabel('РКС/АКС отношение')
    axes[0, 2].set_title('Отношение на мащабните фактори')
    axes[0, 2].grid(True)
    
    # 4. Производни на мащабните фактори
    abs_derivatives = np.gradient(abs_scale_factors, times)
    rel_derivatives = np.gradient(rel_scale_factors, times)
    
    axes[1, 0].plot(times, abs_derivatives, 'b-', label='АКС', marker='o')
    axes[1, 0].plot(times, rel_derivatives, 'r-', label='РКС', marker='s')
    axes[1, 0].set_xlabel('Време (години)')
    axes[1, 0].set_ylabel('da/dt')
    axes[1, 0].set_title('Скорост на разширение')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 5. Корелация между плътност и темп на време
    axes[1, 1].loglog(densities, time_rates, 'purple', marker='d')
    axes[1, 1].set_xlabel('Плътност (kg/m³)')
    axes[1, 1].set_ylabel('Темп на време')
    axes[1, 1].set_title('Корелация плътност-темп на време')
    axes[1, 1].grid(True)
    
    # 6. Логаритмични производни
    log_abs_deriv = np.gradient(np.log(abs_scale_factors), np.log(times))
    log_rel_deriv = np.gradient(np.log(rel_scale_factors), np.log(times))
    
    axes[1, 2].plot(times, log_abs_deriv, 'b-', label='АКС', marker='o')
    axes[1, 2].plot(times, log_rel_deriv, 'r-', label='РКС', marker='s')
    axes[1, 2].set_xlabel('Време (години)')
    axes[1, 2].set_ylabel('d(ln a)/d(ln t)')
    axes[1, 2].set_title('Логаритмични производни')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()

def run_all_examples():
    """
    Изпълнява всички примери последователно
    """
    print("СТАРТИРАНЕ НА ВСИЧКИ ПРИМЕРИ ЗА НЕЛИНЕЙНО ВРЕМЕ КОСМОЛОГИЯ")
    print("=" * 60)
    
    # Пример 1
    params, times = example_1_basic_acs_creation()
    
    # Пример 2
    calculator = example_2_expansion_coefficients()
    
    # Пример 3
    abs_analysis, rel_analysis = example_3_linearity_analysis()
    
    # Пример 4
    abs_coords, rel_coords, expansion_factor = example_4_coordinate_transformation()
    
    # Пример 5
    times, abs_sf, rel_sf, densities, time_rates = example_5_time_evolution()
    
    # Пример 6
    visualizer = example_6_advanced_visualization()
    
    print("\n" + "=" * 60)
    print("ВСИЧКИ ПРИМЕРИ ЗАВЪРШЕНИ УСПЕШНО!")
    
    return {
        'params': params,
        'calculator': calculator,
        'abs_analysis': abs_analysis,
        'rel_analysis': rel_analysis,
        'visualizer': visualizer
    }

if __name__ == "__main__":
    results = run_all_examples() 