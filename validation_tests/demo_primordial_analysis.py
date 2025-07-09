#!/usr/bin/env python3
"""
Демонстрация на анализа на първобитните флуктуации
===============================================

Този скрипт показва как да се използват новите модули за:
1. Анализ на рекомбинацията
2. Анализ на остатъчния шум
3. Комплексен анализ на първобитните флуктуации

Включва размишленията за образуване на атоми по време на рекомбинация
и възможността за едновременно образуване на ранни галактики.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

from primordial_analysis import (
    RecombinationAnalyzer,
    RelicNoiseAnalyzer,
    PrimordialFluctuationAnalyzer
)

# Настройка на логирането
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_recombination_analysis():
    """Демонстрация на анализа на рекомбинацията."""
    print("🔬 ДЕМОНСТРАЦИЯ НА АНАЛИЗА НА РЕКОМБИНАЦИЯТА")
    print("=" * 60)
    
    # Създаване на анализатор
    analyzer = RecombinationAnalyzer(z_recomb=1100)
    
    # Времева ос
    tau = np.linspace(1.0, 4.0, 1000)
    
    # Енергийна плътност
    rho = analyzer.energy_density_evolution(tau)
    
    print(f"τ при рекомбинация: {analyzer.tau_recomb:.3f}")
    print(f"Критична енергийна плътност: {analyzer.rho_recomb_critical:.2e} J/m³")
    
    # Анализ на рекомбинационния прозорец
    analysis = analyzer.plot_recombination_analysis(tau, rho)
    
    # Генериране на доклад
    report = analyzer.generate_comprehensive_report(tau, rho)
    print("\n📄 ДОКЛАД ЗА РЕКОМБИНАЦИЯТА:")
    print(report)
    
    return analysis

def demo_relic_noise_analysis():
    """Демонстрация на анализа на остатъчния шум."""
    print("\n🔊 ДЕМОНСТРАЦИЯ НА АНАЛИЗА НА ОСТАТЪЧНИЯ ШУМ")
    print("=" * 60)
    
    # Създаване на анализатор
    analyzer = RelicNoiseAnalyzer(cmb_amplitude=1e-5)
    
    # Времева ос
    tau = np.linspace(1.0, 6.0, 1000)
    
    # Основна плътност
    rho_base = 1 / (tau**2)
    
    # Генериране на остатъчен шум
    delta_rho = analyzer.generate_primordial_noise(tau, spectral_index=0.96)
    
    print(f"CMB амплитуда: {analyzer.cmb_amplitude:.2e}")
    print(f"Стандартно отклонение на шума: {np.std(delta_rho):.2e}")
    
    # Комплексен анализ
    spectral_analysis, stats = analyzer.plot_comprehensive_analysis(tau, rho_base, delta_rho)
    
    # Генериране на доклад
    report = analyzer.generate_relic_noise_report(tau, rho_base, delta_rho)
    print("\n📄 ДОКЛАД ЗА ОСТАТЪЧНИЯ ШУМ:")
    print(report)
    
    return spectral_analysis, stats

def demo_primordial_fluctuations():
    """Демонстрация на пълния анализ на първобитните флуктуации."""
    print("\n🌌 ДЕМОНСТРАЦИЯ НА АНАЛИЗА НА ПЪРВОБИТНИТЕ ФЛУКТУАЦИИ")
    print("=" * 60)
    
    # Създаване на анализатор
    analyzer = PrimordialFluctuationAnalyzer(z_recomb=1100, cmb_amplitude=1e-5)
    
    # Пълен анализ
    results = analyzer.run_complete_analysis(tau_range=(0.5, 5.0), n_points=1000)
    
    print("\n📊 РЕЗУЛТАТИ ОТ АНАЛИЗА:")
    print("-" * 40)
    
    structure_analysis = results['structure_analysis']
    comparison = results['comparison']
    
    print(f"Фракция колапсиращи региони: {structure_analysis['collapse_fraction']:.2%}")
    print(f"Брой колапсиращи групи: {len(structure_analysis['collapse_groups'])}")
    print(f"Корелация с ΛCDM: {comparison['correlation_rho']:.3f}")
    print(f"Статистическа значимост: {comparison['correlation_p_value']:.3f}")
    
    # Финален доклад
    print("\n📄 ФИНАЛЕН ДОКЛАД:")
    print(results['final_report'])
    
    return results

def demo_theoretical_implications():
    """Демонстрация на теоретичните импликации."""
    print("\n🧠 ТЕОРЕТИЧНИ ИМПЛИКАЦИИ")
    print("=" * 60)
    
    print("""
    📝 КЛЮЧОВИ РАЗМИШЛЕНИЯ:
    
    1. УДЪЛЖЕНА РЕКОМБИНАЦИЯ:
       - В теорията за нелинейно време, рекомбинацията протича по-дълго
       - Това позволява по-стабилно образуване на водородни и хелиеви атоми
       - Удълженият период създава условия за ранно структурообразуване
    
    2. ЛОКАЛНИ ФЛУКТУАЦИИ:
       - Нееднородното разпределение на началните енергии
       - Води до локални региони с различно темпо на рекомбинация
       - Първите "прозрачни" зони могат да станат центрове на галактики
    
    3. ЕДНОВРЕМЕННО ОБРАЗУВАНЕ:
       - Възможност за едновременно образуване на атоми и ранни структури
       - Нарушава традиционната временна последователност
       - Обяснява наблюденията на ранни галактики при високи redshift
    
    4. ОСТАТЪЧЕН ШУМ:
       - Квантовите флуктуации се запазват в CMB
       - Спектралният индекс е съвместим с наблюденията
       - Възможност за детекция на сигнатури от нелинейното време
    
    5. КОСМОЛОГИЧНИ ПОСЛЕДИЦИ:
       - Модифицирани функции на растеж на структурите
       - Различни корелационни функции в CMB
       - Нови прогнози за ранната Вселена
    """)

def main():
    """Основна демонстрационна функция."""
    print("🚀 ДЕМОНСТРАЦИЯ НА ПЪРВОБИТНИЯ АНАЛИЗ")
    print("=" * 80)
    print("Анализ на рекомбинацията, остатъчния шум и първобитните флуктуации")
    print("в контекста на теорията за нелинейно време")
    print("=" * 80)
    
    try:
        # 1. Демонстрация на рекомбинацията
        recomb_analysis = demo_recombination_analysis()
        
        # 2. Демонстрация на остатъчния шум
        noise_analysis = demo_relic_noise_analysis()
        
        # 3. Демонстрация на първобитните флуктуации
        fluct_results = demo_primordial_fluctuations()
        
        # 4. Теоретични импликации
        demo_theoretical_implications()
        
        print("\n✅ ВСИЧКИ ДЕМОНСТРАЦИИ ЗАВЪРШЕНИ УСПЕШНО!")
        
    except Exception as e:
        logger.error(f"Грешка в демонстрацията: {e}")
        print(f"❌ Грешка: {e}")
        raise

if __name__ == "__main__":
    main() 