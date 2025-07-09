"""
Пример за калибриране на скоростта на разширение спрямо червеното отместване
"""

from redshift_calibration import *
import numpy as np
import matplotlib.pyplot as plt

def example_1_basic_calibration():
    """
    Пример 1: Основно калибриране на скоростта на разширение
    """
    print("=== ПРИМЕР 1: ОСНОВНО КАЛИБРИРАНЕ ===")
    
    # Създаваме данни за наблюдения
    obs_data = ObservationalData([], current_age=13.8e9)
    
    # Създаваме калибратор
    calibrator = ExpansionRateCalibrator(obs_data)
    
    # Калибриране
    results = calibrator.calibrate_expansion_rate()
    
    print(f"Оптимална скорост на разширение: {results['optimal_expansion_rate']:.6f}")
    print(f"Крайна грешка: {results['final_error']:.6f}")
    print(f"Успешно: {results['success']}")
    
    # Показваме предсказанията
    print("\nПредсказания:")
    for i, pred in enumerate(results['predictions']):
        print(f"  Точка {i+1}: z_obs={pred['observed_z']:.1f}, z_pred={pred['predicted_z']:.3f}, "
              f"грешка={pred['error']:.3f}, възраст={pred['age']/1e9:.1f} млрд години")
    
    return results

def example_2_linear_time_steps():
    """
    Пример 2: Анализ с линейни времеви стъпки
    """
    print("\n=== ПРИМЕР 2: ЛИНЕЙНИ ВРЕМЕВИ СТЪПКИ ===")
    
    # Различни размери на стъпките за тестване
    step_sizes = [0.5e9, 1e9, 2e9]  # 0.5, 1, 2 млрд години
    
    for step_size in step_sizes:
        print(f"\nАнализ със стъпка от {step_size/1e9:.1f} млрд години:")
        
        # Създаваме генератор на времеви стъпки
        step_generator = LinearTimeStepGenerator(
            start_time=1e9, 
            end_time=13.8e9, 
            step_size=step_size
        )
        
        time_steps = step_generator.get_time_steps()
        print(f"  Брой стъпки: {len(time_steps)}")
        print(f"  Първи 5 стъпки: {[t/1e9 for t in time_steps[:5]]}")
        
        # Калибриране
        obs_data = ObservationalData([], current_age=13.8e9)
        calibrator = ExpansionRateCalibrator(obs_data)
        results = calibrator.calibrate_expansion_rate()
        
        # Създаваме АКС последователност
        params = CosmologicalParameters(linear_expansion_rate=results['optimal_expansion_rate'])
        acs_sequence = step_generator.get_acs_sequence(params)
        
        # Проверяваме линейността
        calculator = ExpansionCalculator(params)
        coefficients = []
        for i in range(len(time_steps) - 1):
            coeff = calculator.calculate_abs_expansion_coefficient(time_steps[i], time_steps[i+1])
            coefficients.append(coeff)
        
        mean_coeff = np.mean(coefficients)
        std_coeff = np.std(coefficients)
        
        print(f"  Среден коефициент на разширение: {mean_coeff:.6f}")
        print(f"  Стандартно отклонение: {std_coeff:.6f}")
        print(f"  Мярка за линейност: {std_coeff/mean_coeff:.6f}")

def example_3_redshift_comparison():
    """
    Пример 3: Сравнение на предсказанията с наблюденията
    """
    print("\n=== ПРИМЕР 3: СРАВНЕНИЕ НА ПРЕДСКАЗАНИЯТА ===")
    
    # Калибриране
    obs_data = ObservationalData([], current_age=13.8e9)
    calibrator = ExpansionRateCalibrator(obs_data)
    results = calibrator.calibrate_expansion_rate()
    
    # Създаваме визуализатор
    visualizer = RedshiftComparisonVisualizer(calibrator)
    
    # Показваме графики
    visualizer.plot_redshift_comparison(results)
    
    # Чувствителност към скоростта на разширение
    visualizer.plot_expansion_rate_sensitivity((0.1, 3.0))
    
    return visualizer

def example_4_custom_redshift_data():
    """
    Пример 4: Работа с персонализирани данни за червено отместване
    """
    print("\n=== ПРИМЕР 4: ПЕРСОНАЛИЗИРАНИ ДАННИ ===")
    
    # Създаваме персонализирани данни
    custom_redshift_data = [
        RedshiftData(0.05, 200, 13.0e9),   # близки галактики
        RedshiftData(0.2, 800, 11.5e9),    # средни разстояния
        RedshiftData(0.8, 3000, 8.0e9),    # далечни галактики
        RedshiftData(1.5, 5000, 6.0e9),    # много далечни галактики
        RedshiftData(3.0, 8000, 3.5e9),    # ранни галактики
        RedshiftData(6.0, 12000, 1.5e9),   # първични галактики
    ]
    
    # Създаваме калибратор с персонализирани данни
    obs_data = ObservationalData(custom_redshift_data, current_age=13.8e9)
    calibrator = ExpansionRateCalibrator(obs_data)
    
    # Предефинираме метода за създаване на данни
    def custom_create_standard_redshift_data():
        return custom_redshift_data
    
    calibrator.create_standard_redshift_data = custom_create_standard_redshift_data
    
    # Калибриране
    results = calibrator.calibrate_expansion_rate()
    
    print(f"Оптимална скорост с персонализирани данни: {results['optimal_expansion_rate']:.6f}")
    print(f"Крайна грешка: {results['final_error']:.6f}")
    
    # Показваме резултатите
    print("\nРезултати с персонализирани данни:")
    for i, pred in enumerate(results['predictions']):
        print(f"  z_obs={pred['observed_z']:.1f}, z_pred={pred['predicted_z']:.3f}, "
              f"грешка={pred['error']:.3f}")
    
    return results

def example_5_sensitivity_analysis():
    """
    Пример 5: Анализ на чувствителността
    """
    print("\n=== ПРИМЕР 5: АНАЛИЗ НА ЧУВСТВИТЕЛНОСТТА ===")
    
    obs_data = ObservationalData([], current_age=13.8e9)
    calibrator = ExpansionRateCalibrator(obs_data)
    
    # Тестваме различни обхвати на скоростта
    rate_ranges = [
        (0.1, 2.0),
        (0.5, 1.5),
        (0.8, 1.2)
    ]
    
    plt.figure(figsize=(15, 5))
    
    for i, rate_range in enumerate(rate_ranges):
        plt.subplot(1, 3, i+1)
        
        rates = np.linspace(rate_range[0], rate_range[1], 100)
        errors = []
        
        redshift_data = calibrator.create_standard_redshift_data()
        
        for rate in rates:
            error = calibrator.objective_function(rate, redshift_data)
            errors.append(error)
        
        plt.plot(rates, errors, 'b-', linewidth=2)
        plt.xlabel('Скорост на разширение k')
        plt.ylabel('Грешка')
        plt.title(f'Обхват {rate_range[0]:.1f} - {rate_range[1]:.1f}')
        plt.grid(True, alpha=0.3)
        
        # Намираме минимума
        min_idx = np.argmin(errors)
        plt.plot(rates[min_idx], errors[min_idx], 'ro', markersize=8)
        plt.text(rates[min_idx], errors[min_idx], f'k={rates[min_idx]:.3f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def example_6_time_evolution_analysis():
    """
    Пример 6: Анализ на времевата еволюция
    """
    print("\n=== ПРИМЕР 6: ВРЕМЕВА ЕВОЛЮЦИЯ ===")
    
    # Калибриране
    obs_data = ObservationalData([], current_age=13.8e9)
    calibrator = ExpansionRateCalibrator(obs_data)
    results = calibrator.calibrate_expansion_rate()
    
    # Създаваме оптимизирани параметри
    params = CosmologicalParameters(linear_expansion_rate=results['optimal_expansion_rate'])
    
    # Времева еволюция
    times = np.linspace(0.5e9, 13.8e9, 50)
    
    scale_factors = []
    densities = []
    time_rates = []
    redshifts = []
    
    redshift_calc = RedshiftCalculator(params)
    
    for t in times:
        acs = AbsoluteCoordinateSystem(t, params)
        
        scale_factors.append(acs.scale_factor)
        densities.append(acs.density)
        time_rates.append(acs.time_rate)
        
        # Червено отместване спрямо текущото време
        z = redshift_calc.calculate_redshift_from_age(t, 13.8e9)
        redshifts.append(z)
    
    # Графики
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Мащабен фактор
    axes[0, 0].plot(times/1e9, scale_factors, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Време (млрд години)')
    axes[0, 0].set_ylabel('Мащабен фактор')
    axes[0, 0].set_title('Еволюция на мащабния фактор')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Плътност
    axes[0, 1].loglog(times/1e9, densities, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Време (млрд години)')
    axes[0, 1].set_ylabel('Плътност (kg/m³)')
    axes[0, 1].set_title('Еволюция на плътността')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Темп на време
    axes[1, 0].semilogx(times/1e9, time_rates, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Време (млрд години)')
    axes[1, 0].set_ylabel('Темп на време')
    axes[1, 0].set_title('Еволюция на темпа на времето')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Червено отместване
    axes[1, 1].semilogy(times/1e9, redshifts, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Време (млрд години)')
    axes[1, 1].set_ylabel('Червено отместване z')
    axes[1, 1].set_title('Еволюция на червеното отместване')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Използвана оптимална скорост: {results['optimal_expansion_rate']:.6f}")
    print(f"Мащабен фактор в началото: {scale_factors[0]:.2e}")
    print(f"Мащабен фактор в края: {scale_factors[-1]:.2e}")
    print(f"Максимално червено отместване: {max(redshifts):.2f}")

def run_all_redshift_examples():
    """
    Изпълнява всички примери за калибриране на червеното отместване
    """
    print("СТАРТИРАНЕ НА ВСИЧКИ ПРИМЕРИ ЗА КАЛИБРИРАНЕ НА ЧЕРВЕНОТО ОТМЕСТВАНЕ")
    print("=" * 70)
    
    # Пример 1
    basic_results = example_1_basic_calibration()
    
    # Пример 2
    example_2_linear_time_steps()
    
    # Пример 3
    visualizer = example_3_redshift_comparison()
    
    # Пример 4
    custom_results = example_4_custom_redshift_data()
    
    # Пример 5
    example_5_sensitivity_analysis()
    
    # Пример 6
    example_6_time_evolution_analysis()
    
    print("\n" + "=" * 70)
    print("ВСИЧКИ ПРИМЕРИ ЗА КАЛИБРИРАНЕ ЗАВЪРШЕНИ УСПЕШНО!")
    
    return {
        'basic_results': basic_results,
        'custom_results': custom_results,
        'visualizer': visualizer
    }

if __name__ == "__main__":
    results = run_all_redshift_examples() 