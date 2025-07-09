#!/usr/bin/env python3
"""
Пример за използване на модула за АКС времева трансформация
Демонстрира теоретичните разработки за линейно разширение в абсолютно време
"""

import numpy as np
import matplotlib.pyplot as plt
from acs_time_transformation import (
    TimeTransformationModel, RedshiftTimeRelation, 
    ExpansionAnalyzer, ExpansionVisualizer
)

def example_1_basic_transformation():
    """
    Основен пример на времевата трансформация
    """
    print("=== Пример 1: Основна времева трансформация ===")
    
    # Инициализация
    time_model = TimeTransformationModel(k_expansion=1e-3, t_universe_gyr=13.8)
    redshift_model = RedshiftTimeRelation(H0=70)
    analyzer = ExpansionAnalyzer(time_model, redshift_model)
    
    # Генериране на времеви интервали
    t_abs_array = analyzer.generate_discrete_timeline(delta_t_gyr=1, max_t_gyr=13.8)
    
    # Изчисляване на резултатите
    results = analyzer.compute_expansion_table(t_abs_array)
    
    # Печатане на таблицата
    analyzer.print_expansion_table(results)
    
    # Визуализация
    visualizer = ExpansionVisualizer(results)
    visualizer.plot_time_transformation()
    
    print("✅ Основен пример завършен")

def example_2_redshift_analysis():
    """
    Анализ на червеното отместване в различни модели
    """
    print("\n=== Пример 2: Анализ на червеното отместване ===")
    
    # Инициализация
    time_model = TimeTransformationModel(k_expansion=1e-3, t_universe_gyr=13.8)
    redshift_model = RedshiftTimeRelation(H0=70)
    analyzer = ExpansionAnalyzer(time_model, redshift_model)
    
    # Генериране на времеви интервали
    t_abs_array = analyzer.generate_discrete_timeline(delta_t_gyr=0.5, max_t_gyr=13.8)
    results = analyzer.compute_expansion_table(t_abs_array)
    
    # Сравнение на моделите
    visualizer = ExpansionVisualizer(results)
    visualizer.plot_comparison_models()
    
    # Анализ на ключови епохи
    print("\n🔍 Анализ на ключови епохи:")
    key_epochs = [
        (0.1, "Ранна Вселена"),
        (1.0, "Първи млрд. години"),
        (5.0, "Средни епохи"),
        (10.0, "Формиране на галактики"),
        (13.8, "Днес")
    ]
    
    for t_abs, description in key_epochs:
        t_rel = time_model.compute_relative_time(np.array([t_abs]))[0]
        t_rel_norm = t_rel / time_model.compute_relative_time(np.array([13.8]))[0] * 13.8
        a_abs = time_model.scale_factor_absolute(t_abs)
        z = redshift_model.redshift_from_time(np.array([t_abs * 1e9 * 3.1536e16]))[0]
        T_z = time_model.time_transformation_factor(z)
        
        print(f"{description:<20}: t_abs={t_abs:4.1f} Gyr, t_rel={t_rel_norm:5.2f} Gyr, z={z:6.3f}, T(z)={T_z:6.3f}")
    
    print("✅ Анализ на червеното отместване завършен")

def example_3_custom_parameters():
    """
    Пример с различни параметри на модела
    """
    print("\n=== Пример 3: Различни параметри на модела ===")
    
    # Сравнение на различни стойности на k
    k_values = [0.5e-3, 1e-3, 2e-3]
    colors = ['blue', 'red', 'green']
    
    plt.figure(figsize=(12, 8))
    
    for i, k in enumerate(k_values):
        time_model = TimeTransformationModel(k_expansion=k, t_universe_gyr=13.8)
        redshift_model = RedshiftTimeRelation(H0=70)
        analyzer = ExpansionAnalyzer(time_model, redshift_model)
        
        t_abs_array = analyzer.generate_discrete_timeline(delta_t_gyr=0.5, max_t_gyr=13.8)
        results = analyzer.compute_expansion_table(t_abs_array)
        
        plt.subplot(2, 2, i+1)
        plt.plot(results['t_abs_gyr'], results['a_abs'], 
                color=colors[i], linewidth=2, label=f'k = {k:.1e}')
        plt.xlabel('Абсолютно време [Gyr]')
        plt.ylabel('Мащабен фактор a(t_abs)')
        plt.title(f'Линейно разширение с k = {k:.1e}')
        plt.grid(True)
        plt.legend()
    
    # Сравнение на всички модели
    plt.subplot(2, 2, 4)
    for i, k in enumerate(k_values):
        time_model = TimeTransformationModel(k_expansion=k, t_universe_gyr=13.8)
        redshift_model = RedshiftTimeRelation(H0=70)
        analyzer = ExpansionAnalyzer(time_model, redshift_model)
        
        t_abs_array = analyzer.generate_discrete_timeline(delta_t_gyr=0.5, max_t_gyr=13.8)
        results = analyzer.compute_expansion_table(t_abs_array)
        
        plt.plot(results['t_abs_gyr'], results['a_abs'], 
                color=colors[i], linewidth=2, label=f'k = {k:.1e}')
    
    plt.xlabel('Абсолютно време [Gyr]')
    plt.ylabel('Мащабен фактор a(t_abs)')
    plt.title('Сравнение на различни стойности на k')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("✅ Анализ с различни параметри завършен")

def example_4_time_dilation_analysis():
    """
    Подробен анализ на времевата дилатация
    """
    print("\n=== Пример 4: Анализ на времевата дилатация ===")
    
    # Инициализация
    time_model = TimeTransformationModel(k_expansion=1e-3, t_universe_gyr=13.8)
    
    # Диапазон от червени отмествания
    z_range = np.logspace(-3, 2, 100)  # от z=0.001 до z=100
    
    # Изчисляване на различни трансформационни фактори
    T_z = time_model.time_transformation_factor(z_range)
    density_factor = time_model.density_approximation(z_range)
    
    # Графики
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: Времевия трансформационен фактор
    axes[0, 0].loglog(z_range, T_z, 'b-', linewidth=2, label='T(z) = 1/(1+z)^(3/2)')
    axes[0, 0].set_xlabel('Червено отместване z')
    axes[0, 0].set_ylabel('Времевия фактор T(z)')
    axes[0, 0].set_title('Времева трансформация')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # График 2: Плътност на материята
    axes[0, 1].loglog(z_range, density_factor, 'r-', linewidth=2, label='ρ(z) ∝ (1+z)³')
    axes[0, 1].set_xlabel('Червено отместване z')
    axes[0, 1].set_ylabel('Плътност ρ(z)')
    axes[0, 1].set_title('Плътност на материята')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # График 3: Обратна зависимост
    axes[1, 0].loglog(z_range, 1/T_z, 'g-', linewidth=2, label='1/T(z) = (1+z)^(3/2)')
    axes[1, 0].set_xlabel('Червено отместване z')
    axes[1, 0].set_ylabel('1/T(z)')
    axes[1, 0].set_title('Ефект на забавяне на времето')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # График 4: Сравнение на времевите ефекти
    axes[1, 1].loglog(z_range, T_z, 'b-', linewidth=2, label='T(z) = 1/(1+z)^(3/2)')
    axes[1, 1].loglog(z_range, 1/(1+z_range), 'r--', linewidth=2, label='1/(1+z) [стандартен]')
    axes[1, 1].set_xlabel('Червено отместване z')
    axes[1, 1].set_ylabel('Времевия фактор')
    axes[1, 1].set_title('Сравнение на времевите ефекти')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Численни примери
    print("\n📊 Ключови стойности на времевата дилатация:")
    test_z_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
    
    print(f"{'z':<8} {'T(z)':<12} {'1/T(z)':<12} {'Забавяне':<15}")
    print("-" * 50)
    
    for z in test_z_values:
        T_val = time_model.time_transformation_factor(z)
        slowdown = 1/T_val
        print(f"{z:<8.1f} {T_val:<12.6f} {slowdown:<12.2f} {slowdown:.1f}x по-бавно")
    
    print("✅ Анализ на времевата дилатация завършен")

def example_5_comprehensive_analysis():
    """
    Цялостен анализ на модела
    """
    print("\n=== Пример 5: Цялостен анализ на модела ===")
    
    # Инициализация
    time_model = TimeTransformationModel(k_expansion=1e-3, t_universe_gyr=13.8)
    redshift_model = RedshiftTimeRelation(H0=70)
    analyzer = ExpansionAnalyzer(time_model, redshift_model)
    
    # Различни дискретизации
    discretizations = [0.5, 1.0, 2.0]
    
    print("📊 Сравнение на различни дискретизации:")
    
    for delta_t in discretizations:
        print(f"\n--- Дискретизация: {delta_t} Gyr ---")
        
        t_abs_array = analyzer.generate_discrete_timeline(delta_t_gyr=delta_t, max_t_gyr=13.8)
        results = analyzer.compute_expansion_table(t_abs_array)
        
        # Статистики
        print(f"Брой точки: {len(results['t_abs_gyr'])}")
        print(f"Диапазон z: {np.min(results['z_values']):.3f} - {np.max(results['z_values']):.3f}")
        print(f"Диапазон T(z): {np.min(results['time_transform_factor']):.6f} - {np.max(results['time_transform_factor']):.6f}")
        
        # Средни стойности
        mean_expansion = np.mean(results['a_abs'])
        mean_z = np.mean(results['z_values'])
        
        print(f"Средно разширение: {mean_expansion:.6f}")
        print(f"Средно червено отместване: {mean_z:.3f}")
    
    # Създаване на финалната визуализация
    t_abs_array = analyzer.generate_discrete_timeline(delta_t_gyr=1.0, max_t_gyr=13.8)
    results = analyzer.compute_expansion_table(t_abs_array)
    
    visualizer = ExpansionVisualizer(results)
    visualizer.plot_time_transformation()
    visualizer.plot_comparison_models()
    
    print("\n✅ Цялостен анализ завършен")

def main():
    """
    Стартиране на всички примери
    """
    print("🌌 ПРИМЕРИ ЗА АКС ВРЕМЕВА ТРАНСФОРМАЦИЯ")
    print("=" * 60)
    
    # Стартиране на примерите
    example_1_basic_transformation()
    example_2_redshift_analysis()
    example_3_custom_parameters()
    example_4_time_dilation_analysis()
    example_5_comprehensive_analysis()
    
    print("\n🎯 ЗАКЛЮЧЕНИЯ:")
    print("• Линейното разширение в АКС води до нелинейни ефекти в РКС")
    print("• Времевата трансформация T(z) = 1/(1+z)^(3/2) обяснява космическото ускорение")
    print("• Компресираната хронология е естествено следствие от високата енергийна плътност")
    print("• Моделът предлага алтернатива на тъмната енергия")
    
    print("\n🚀 Всички примери завършени успешно!")

if __name__ == "__main__":
    main() 