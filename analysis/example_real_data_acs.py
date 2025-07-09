#!/usr/bin/env python3
"""
Пример за анализ на реални данни за АКС с Pantheon+ данни
Демонстрира как да се намери единна АКС и да се генерират линейни интервали
"""

import numpy as np
import matplotlib.pyplot as plt
from real_data_acs_analysis import (
    PantheonDataLoader, UnifiedACSFinder, LinearACSGenerator,
    LinearExpansionAnalyzer, RealDataACSVisualizer
)

def example_1_basic_analysis():
    """Основен анализ с реални данни от Pantheon+"""
    
    print("=== Пример 1: Основен анализ с реални данни ===")
    
    # Зареждаме данните
    loader = PantheonDataLoader()
    if not loader.load_data():
        print("Неуспешно зареждане на данните")
        return
    
    # Извличаме данни за червено отместване
    redshift_data, distance_data = loader.get_redshift_data(max_redshift=1.5)
    print(f"Заредени {len(redshift_data)} записа за анализ")
    
    # Намираме единна АКС
    acs_finder = UnifiedACSFinder(age_universe=13.8e9)
    unified_acs = acs_finder.find_unified_acs(redshift_data, distance_data)
    
    if unified_acs:
        print(f"Единна АКС: възраст = {unified_acs['age']/1e9:.1f} млрд. години")
        print(f"Червено отместване: z = {unified_acs['redshift']:.3f}")
        
        # Генерираме линейни интервали
        linear_generator = LinearACSGenerator(unified_acs)
        linear_acs_systems = linear_generator.generate_linear_intervals(
            num_intervals=5, interval_size=2.5e9
        )
        
        print("\nАКС системи с линейни интервали:")
        for i, acs in enumerate(linear_acs_systems):
            print(f"  АКС {i+1}: {acs['age']/1e9:.1f} млрд. г., "
                  f"z = {acs['redshift']:.3f}, коефициент = {acs['expansion_factor']:.3f}")

def example_2_detailed_analysis():
    """Подробен анализ с различни параметри"""
    
    print("\n=== Пример 2: Подробен анализ с различни параметри ===")
    
    # Зареждаме данните
    loader = PantheonDataLoader()
    if not loader.load_data():
        return
    
    # Извличаме данни с различни граници
    redshift_data, distance_data = loader.get_redshift_data(max_redshift=2.0)
    
    # Намираме единна АКС
    acs_finder = UnifiedACSFinder(age_universe=13.8e9)
    unified_acs = acs_finder.find_unified_acs(redshift_data, distance_data)
    
    if unified_acs:
        # Генерираме по-много АКС системи
        linear_generator = LinearACSGenerator(unified_acs)
        linear_acs_systems = linear_generator.generate_linear_intervals(
            num_intervals=8, interval_size=1.5e9
        )
        
        # Анализираме разширението
        analyzer = LinearExpansionAnalyzer(unified_acs, linear_acs_systems)
        expansion_coeffs = analyzer.calculate_expansion_coefficients()
        
        print(f"\nГенерирани {len(linear_acs_systems)} АКС системи:")
        for coeff in expansion_coeffs:
            print(f"  АКС {coeff['acs_index']+1}: възраст = {coeff['age']/1e9:.1f} млрд. г., "
                  f"z = {coeff['redshift']:.3f}, коефициент = {coeff['expansion_coefficient']:.3f}")
        
        # Сравняваме с наблюденията
        theoretical_data = analyzer.compare_with_observations(redshift_data, distance_data)
        
        print(f"\nТеоретични стойности за линейно разширение:")
        for i, data in enumerate(theoretical_data[:5]):  # Показваме първите 5
            print(f"  z = {data['redshift']:.3f}, μ_теор = {data['theoretical_distance']:.2f}")

def example_3_visualization():
    """Визуализация на резултатите"""
    
    print("\n=== Пример 3: Визуализация на резултатите ===")
    
    # Зареждаме данните
    loader = PantheonDataLoader()
    if not loader.load_data():
        return
    
    redshift_data, distance_data = loader.get_redshift_data(max_redshift=1.8)
    
    # Намираме единна АКС
    acs_finder = UnifiedACSFinder(age_universe=13.8e9)
    unified_acs = acs_finder.find_unified_acs(redshift_data, distance_data)
    
    if unified_acs:
        # Генерираме АКС системи
        linear_generator = LinearACSGenerator(unified_acs)
        linear_acs_systems = linear_generator.generate_linear_intervals(
            num_intervals=6, interval_size=2.0e9
        )
        
        # Визуализираме резултатите
        visualizer = RealDataACSVisualizer()
        
        # Основен анализ
        visualizer.plot_unified_acs_analysis(unified_acs, linear_acs_systems,
                                            redshift_data, distance_data)
        
        # Времева линия
        visualizer.plot_acs_timeline(unified_acs, linear_acs_systems)
        
        print("Визуализацията е завършена")

def example_4_comparison_study():
    """Сравнително изследване на различни интервали"""
    
    print("\n=== Пример 4: Сравнително изследване ===")
    
    # Зареждаме данните
    loader = PantheonDataLoader()
    if not loader.load_data():
        return
    
    redshift_data, distance_data = loader.get_redshift_data(max_redshift=1.5)
    
    # Намираме единна АКС
    acs_finder = UnifiedACSFinder(age_universe=13.8e9)
    unified_acs = acs_finder.find_unified_acs(redshift_data, distance_data)
    
    if unified_acs:
        # Тестваме различни размери на интервали
        interval_sizes = [1.0e9, 1.5e9, 2.0e9, 2.5e9, 3.0e9]
        
        print("\nСравнение на различни размери на интервали:")
        for size in interval_sizes:
            linear_generator = LinearACSGenerator(unified_acs)
            linear_acs_systems = linear_generator.generate_linear_intervals(
                num_intervals=5, interval_size=size
            )
            
            # Изчисляваме средния коефициент на разширение
            avg_expansion = np.mean([acs['expansion_factor'] for acs in linear_acs_systems])
            
            print(f"  Интервал {size/1e9:.1f} млрд. г.: "
                  f"{len(linear_acs_systems)} АКС, среден коефициент = {avg_expansion:.3f}")

def example_5_redshift_calibration():
    """Калибриране на червено отместване с реални данни"""
    
    print("\n=== Пример 5: Калибриране на червено отместване ===")
    
    # Зареждаме данните
    loader = PantheonDataLoader()
    if not loader.load_data():
        return
    
    redshift_data, distance_data = loader.get_redshift_data(max_redshift=1.2)
    
    # Намираме единна АКС
    acs_finder = UnifiedACSFinder(age_universe=13.8e9)
    unified_acs = acs_finder.find_unified_acs(redshift_data, distance_data)
    
    if unified_acs:
        # Генерираме АКС системи
        linear_generator = LinearACSGenerator(unified_acs)
        linear_acs_systems = linear_generator.generate_linear_intervals(
            num_intervals=6, interval_size=2.0e9
        )
        
        # Анализираме разширението
        analyzer = LinearExpansionAnalyzer(unified_acs, linear_acs_systems)
        
        # Показваме калибрационни данни
        print("\nКалибрационни данни за червено отместване:")
        print("Възраст (млрд. г.) | Червено отместване | Коефициент")
        print("-" * 55)
        
        for acs in linear_acs_systems:
            print(f"{acs['age']/1e9:15.1f} | {acs['redshift']:17.3f} | {acs['expansion_factor']:10.3f}")
        
        # Демонстрираме формулата за преобразуване
        print(f"\nФормула за преобразуване възраст -> червено отместване:")
        print(f"z = (2/3 * H0^-1 * 9.777e11 / age)^(2/3) - 1")
        print(f"където H0 = 70 km/s/Mpc")

def example_6_advanced_analysis():
    """Напреднал анализ с оптимизация"""
    
    print("\n=== Пример 6: Напреднал анализ с оптимизация ===")
    
    # Зареждаме данните
    loader = PantheonDataLoader()
    if not loader.load_data():
        return
    
    redshift_data, distance_data = loader.get_redshift_data(max_redshift=2.0)
    
    # Намираме единна АКС
    acs_finder = UnifiedACSFinder(age_universe=13.8e9)
    unified_acs = acs_finder.find_unified_acs(redshift_data, distance_data)
    
    if unified_acs:
        # Оптимизираме размера на интервала
        best_interval = None
        best_error = float('inf')
        
        for interval_size in np.linspace(1.0e9, 3.0e9, 10):
            linear_generator = LinearACSGenerator(unified_acs)
            linear_acs_systems = linear_generator.generate_linear_intervals(
                num_intervals=6, interval_size=interval_size
            )
            
            # Изчисляваме грешка спрямо наблюденията
            analyzer = LinearExpansionAnalyzer(unified_acs, linear_acs_systems)
            theoretical_data = analyzer.compare_with_observations(redshift_data, distance_data)
            
            # Опростена грешка - може да се подобри
            error = np.sum([abs(data['redshift'] - 0.5) for data in theoretical_data if data['redshift'] > 0])
            
            if error < best_error:
                best_error = error
                best_interval = interval_size
        
        print(f"Оптимален размер на интервала: {best_interval/1e9:.2f} млрд. години")
        print(f"Грешка: {best_error:.3f}")
        
        # Генерираме с оптималния интервал
        linear_generator = LinearACSGenerator(unified_acs)
        linear_acs_systems = linear_generator.generate_linear_intervals(
            num_intervals=6, interval_size=best_interval
        )
        
        print(f"\nОптимални АКС системи:")
        for i, acs in enumerate(linear_acs_systems):
            print(f"  АКС {i+1}: {acs['age']/1e9:.1f} млрд. г., "
                  f"z = {acs['redshift']:.3f}, коефициент = {acs['expansion_factor']:.3f}")

def main():
    """Изпълнява всички примери"""
    
    print("=== Анализ на реални данни за АКС с Pantheon+ ===")
    print("Този анализ използва реални данни от Pantheon+ за намиране на")
    print("единна АКС и генериране на линейни интервали.")
    print()
    
    try:
        example_1_basic_analysis()
        example_2_detailed_analysis()
        example_3_visualization()
        example_4_comparison_study()
        example_5_redshift_calibration()
        example_6_advanced_analysis()
        
        print("\n=== Всички примери завършени успешно ===")
        
    except Exception as e:
        print(f"\nГрешка при изпълнение: {e}")
        print("Моля, проверете дали папката с данни е достъпна.")

if __name__ == "__main__":
    main() 