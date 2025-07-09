#!/usr/bin/env python3
"""
Бърз тест за модула с реални данни за АКС
"""

import numpy as np
import sys
import os

def test_import():
    """Тест за импортиране на модула"""
    try:
        from real_data_acs_analysis import (
            PantheonDataLoader, UnifiedACSFinder, LinearACSGenerator,
            LinearExpansionAnalyzer, RealDataACSVisualizer
        )
        print("✓ Модулът се импортира успешно")
        return True
    except Exception as e:
        print(f"✗ Грешка при импортиране: {e}")
        return False

def test_data_loading():
    """Тест за зареждане на данни"""
    try:
        from real_data_acs_analysis import PantheonDataLoader
        
        # Зареждане на реални данни
        data_path = r"D:\MyPRJ\Python\NotLinearTime\test_2\data\Pantheon+_Data\4_DISTANCES_AND_COVAR\Pantheon+SH0ES.dat"
        if os.path.exists(data_path):
            loader = PantheonDataLoader(data_path)
            if loader.load_data():
                print("✓ Данните се зареждат успешно")
                return True
            else:
                print("✗ Грешка при зареждане на данните")
                return False
        else:
            print("⚠ Файлът с данни не е намерен - използваме симулирани данни")
            return test_simulated_data()
            
    except Exception as e:
        print(f"✗ Грешка при тест на зареждане: {e}")
        return False

def test_simulated_data():
    """Тест със симулирани данни"""
    try:
        from real_data_acs_analysis import UnifiedACSFinder, LinearACSGenerator
        
        # Симулираме данни
        redshift_data = np.linspace(0.01, 1.5, 100)
        distance_data = 35 + 5 * np.log10(redshift_data * 3000)  # Опростена формула
        
        # Тестваме намиране на единна АКС
        acs_finder = UnifiedACSFinder(age_universe=13.8e9)
        unified_acs = acs_finder.find_unified_acs(redshift_data, distance_data)
        
        if unified_acs:
            print("✓ Единна АКС се установява успешно")
            
            # Тестваме генериране на линейни интервали
            linear_generator = LinearACSGenerator(unified_acs)
            linear_acs_systems = linear_generator.generate_linear_intervals(
                num_intervals=5, interval_size=2.0e9
            )
            
            if len(linear_acs_systems) > 0:
                print("✓ АКС системи се генерират успешно")
                return True
            else:
                print("✗ Неуспешно генериране на АКС системи")
                return False
        else:
            print("✗ Неуспешно установяване на единна АКС")
            return False
            
    except Exception as e:
        print(f"✗ Грешка при тест със симулирани данни: {e}")
        return False

def test_analysis_functionality():
    """Тест на функционалността за анализ"""
    try:
        from real_data_acs_analysis import (
            UnifiedACSFinder, LinearACSGenerator, LinearExpansionAnalyzer
        )
        
        # Симулираме данни
        redshift_data = np.linspace(0.01, 1.0, 50)
        distance_data = 35 + 5 * np.log10(redshift_data * 3000)
        
        # Намираме единна АКС
        acs_finder = UnifiedACSFinder(age_universe=13.8e9)
        unified_acs = acs_finder.find_unified_acs(redshift_data, distance_data)
        
        if unified_acs:
            # Генерираме АКС системи
            linear_generator = LinearACSGenerator(unified_acs)
            linear_acs_systems = linear_generator.generate_linear_intervals(
                num_intervals=4, interval_size=2.5e9
            )
            
            # Анализираме разширението
            analyzer = LinearExpansionAnalyzer(unified_acs, linear_acs_systems)
            expansion_coeffs = analyzer.calculate_expansion_coefficients()
            
            if len(expansion_coeffs) > 0:
                print("✓ Анализът на разширение работи успешно")
                
                # Тестваме сравнение с наблюденията
                theoretical_data = analyzer.compare_with_observations(redshift_data, distance_data)
                
                if len(theoretical_data) > 0:
                    print("✓ Сравнението с наблюденията работи успешно")
                    return True
                else:
                    print("✗ Грешка при сравнение с наблюденията")
                    return False
            else:
                print("✗ Грешка при анализ на разширение")
                return False
        else:
            print("✗ Неуспешно установяване на единна АКС за анализ")
            return False
            
    except Exception as e:
        print(f"✗ Грешка при тест на анализ: {e}")
        return False

def test_calculations():
    """Тест на изчисленията"""
    try:
        from real_data_acs_analysis import UnifiedACSFinder
        
        acs_finder = UnifiedACSFinder(age_universe=13.8e9)
        
        # Тестваме преобразуване редшифт -> възраст
        test_z = 1.0
        age = acs_finder.redshift_to_age(test_z)
        
        if age > 0 and age < 13.8e9:
            print("✓ Преобразуване redshift -> age работи")
            
            # Тестваме обратното преобразуване
            z_back = acs_finder.age_to_redshift(age)
            
            if abs(z_back - test_z) < 0.1:  # Допустима грешка
                print("✓ Обратното преобразуване age -> redshift работи")
                return True
            else:
                print(f"✗ Грешка в обратното преобразуване: {z_back} vs {test_z}")
                return False
        else:
            print(f"✗ Неправилна възраст: {age}")
            return False
            
    except Exception as e:
        print(f"✗ Грешка при тест на изчисления: {e}")
        return False

def test_visualization():
    """Тест на визуализацията (без показване)"""
    try:
        from real_data_acs_analysis import RealDataACSVisualizer
        
        # Създаваме визуализатор
        visualizer = RealDataACSVisualizer()
        
        if visualizer.fig_size == (12, 8):
            print("✓ Визуализаторът се създава успешно")
            return True
        else:
            print("✗ Грешка при създаване на визуализатор")
            return False
            
    except Exception as e:
        print(f"✗ Грешка при тест на визуализация: {e}")
        return False

def run_all_tests():
    """Изпълнява всички тестове"""
    print("=== Тестване на модула за реални данни за АКС ===")
    print()
    
    tests = [
        ("Импортиране на модула", test_import),
        ("Зареждане на данни", test_data_loading),
        ("Функционалност за анализ", test_analysis_functionality),
        ("Изчисления", test_calculations),
        ("Визуализация", test_visualization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Тестване: {test_name}")
        if test_func():
            passed += 1
        print()
    
    print(f"=== Резултат: {passed}/{total} тестове преминали ===")
    
    if passed == total:
        print("✓ Всички тестове са успешни!")
        return True
    else:
        print("⚠ Някои тестове не са успешни")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 