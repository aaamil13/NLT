#!/usr/bin/env python3
"""
Бърз тест за модула за АКС времева трансформация
"""

import numpy as np
import sys
import os

def test_import():
    """
    Тест за импортиране на модула
    """
    try:
        from acs_time_transformation import (
            TimeTransformationModel, RedshiftTimeRelation, 
            ExpansionAnalyzer, ExpansionVisualizer
        )
        print("✓ Модулът се импортира успешно")
        return True
    except Exception as e:
        print(f"✗ Грешка при импортиране: {e}")
        return False

def test_time_transformation_model():
    """
    Тест за TimeTransformationModel
    """
    try:
        from acs_time_transformation import TimeTransformationModel
        
        # Инициализация
        model = TimeTransformationModel(k_expansion=1e-3, t_universe_gyr=13.8)
        
        # Тестване на методи
        z_test = np.array([0.1, 1.0, 5.0])
        T_z = model.time_transformation_factor(z_test)
        density = model.density_approximation(z_test)
        
        t_abs_test = np.array([1.0, 5.0, 10.0])
        t_rel = model.compute_relative_time(t_abs_test)
        a_abs = model.scale_factor_absolute(t_abs_test)
        a_rel = model.scale_factor_relative(t_rel)
        
        print("✓ TimeTransformationModel работи правилно")
        print(f"  T(z) примери: {T_z}")
        print(f"  Плътност примери: {density}")
        print(f"  Релативно време: {t_rel}")
        return True
    except Exception as e:
        print(f"✗ Грешка в TimeTransformationModel: {e}")
        return False

def test_redshift_time_relation():
    """
    Тест за RedshiftTimeRelation
    """
    try:
        from acs_time_transformation import RedshiftTimeRelation
        
        # Инициализация
        redshift_model = RedshiftTimeRelation(H0=70)
        
        # Тестване на методи
        z_test = np.array([0.1, 1.0, 2.0])
        H_z = redshift_model.hubble_parameter(z_test)
        dt_dz = redshift_model.dt_abs_dz(z_test)
        
        print("✓ RedshiftTimeRelation работи правилно")
        print(f"  H(z) примери: {H_z}")
        print(f"  dt/dz примери: {dt_dz}")
        return True
    except Exception as e:
        print(f"✗ Грешка в RedshiftTimeRelation: {e}")
        return False

def test_expansion_analyzer():
    """
    Тест за ExpansionAnalyzer
    """
    try:
        from acs_time_transformation import (
            TimeTransformationModel, RedshiftTimeRelation, ExpansionAnalyzer
        )
        
        # Инициализация
        time_model = TimeTransformationModel(k_expansion=1e-3, t_universe_gyr=13.8)
        redshift_model = RedshiftTimeRelation(H0=70)
        analyzer = ExpansionAnalyzer(time_model, redshift_model)
        
        # Тестване на методи
        t_abs_array = analyzer.generate_discrete_timeline(delta_t_gyr=2, max_t_gyr=10)
        results = analyzer.compute_expansion_table(t_abs_array)
        
        print("✓ ExpansionAnalyzer работи правилно")
        print(f"  Брой точки: {len(results['t_abs_gyr'])}")
        print(f"  Диапазон z: {np.min(results['z_values']):.3f} - {np.max(results['z_values']):.3f}")
        return True
    except Exception as e:
        print(f"✗ Грешка в ExpansionAnalyzer: {e}")
        return False

def test_expansion_visualizer():
    """
    Тест за ExpansionVisualizer
    """
    try:
        from acs_time_transformation import (
            TimeTransformationModel, RedshiftTimeRelation, 
            ExpansionAnalyzer, ExpansionVisualizer
        )
        
        # Инициализация
        time_model = TimeTransformationModel(k_expansion=1e-3, t_universe_gyr=13.8)
        redshift_model = RedshiftTimeRelation(H0=70)
        analyzer = ExpansionAnalyzer(time_model, redshift_model)
        
        # Генериране на данни
        t_abs_array = analyzer.generate_discrete_timeline(delta_t_gyr=2, max_t_gyr=10)
        results = analyzer.compute_expansion_table(t_abs_array)
        
        # Създаване на визуализатор
        visualizer = ExpansionVisualizer(results)
        
        print("✓ ExpansionVisualizer се инициализира правилно")
        print(f"  Резултати заредени: {len(results['t_abs_gyr'])} точки")
        return True
    except Exception as e:
        print(f"✗ Грешка в ExpansionVisualizer: {e}")
        return False

def test_mathematical_consistency():
    """
    Тест за математическа консистентност
    """
    try:
        from acs_time_transformation import TimeTransformationModel
        
        model = TimeTransformationModel(k_expansion=1e-3, t_universe_gyr=13.8)
        
        # Тестване на консистентност
        z = 1.0
        T_z = model.time_transformation_factor(z)
        density = model.density_approximation(z)
        
        # Проверка: T(z) = 1/(1+z)^(3/2)
        expected_T = 1.0 / (1 + z)**(3/2)
        
        # Проверка: ρ(z) = (1+z)³
        expected_density = (1 + z)**3
        
        if abs(T_z - expected_T) < 1e-10:
            print("✓ Времевата трансформация е математически правилна")
        else:
            print(f"✗ Грешка в времевата трансформация: {T_z} vs {expected_T}")
            return False
        
        if abs(density - expected_density) < 1e-10:
            print("✓ Плътността е математически правилна")
        else:
            print(f"✗ Грешка в плътността: {density} vs {expected_density}")
            return False
        
        # Тестване на интегрирането
        t_abs = 5.0
        t_rel = model.compute_relative_time(np.array([t_abs]))[0]
        expected_t_rel = (2/5) * t_abs**(5/2)
        
        if abs(t_rel - expected_t_rel) < 1e-10:
            print("✓ Интегрирането за релативното време е правилно")
        else:
            print(f"✗ Грешка в интегрирането: {t_rel} vs {expected_t_rel}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Грешка в математическата консистентност: {e}")
        return False

def test_main_function():
    """
    Тест за основната функция
    """
    try:
        from acs_time_transformation import main
        
        print("✓ Основната функция се импортира успешно")
        print("  (Не се изпълнява за да се избегне показването на графики)")
        return True
    except Exception as e:
        print(f"✗ Грешка в основната функция: {e}")
        return False

def main():
    """
    Изпълнение на всички тестове
    """
    print("🧪 ТЕСТВАНЕ НА МОДУЛА ЗА АКС ВРЕМЕВА ТРАНСФОРМАЦИЯ")
    print("=" * 60)
    
    tests = [
        test_import,
        test_time_transformation_model,
        test_redshift_time_relation,
        test_expansion_analyzer,
        test_expansion_visualizer,
        test_mathematical_consistency,
        test_main_function
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        if test():
            passed += 1
        else:
            print(f"НЕУСПЕШЕН ТЕСТ: {test.__name__}")
    
    print("\n" + "=" * 60)
    print(f"РЕЗУЛТАТИ: {passed}/{total} тестове преминаха успешно")
    
    if passed == total:
        print("🎉 Всички тестове са успешни!")
        return True
    else:
        print("❌ Някои тестове са неуспешни!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 