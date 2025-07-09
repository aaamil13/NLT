"""
Обширен анализ на разширените аналитични функции
=================================================

Този скрипт демонстрира всички нови аналитични функции:
1. T(z) - интегрална и приближена форма
2. a(t_abs) - мащабен фактор като функция от абсолютното време
3. H(t_abs) - параметър на Hubble като функция от времето
4. Натурална метрична трансформация
5. Разширен анализ до z > 2

Автор: Система за анализ на нелинейно време
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

from lib.advanced_analytical_functions import AdvancedAnalyticalFunctions, create_analytical_functions, quick_t_z_analysis


def main():
    print("=" * 80)
    print("ОБШИРЕН АНАЛИЗ НА РАЗШИРЕНИТЕ АНАЛИТИЧНИ ФУНКЦИИ")
    print("=" * 80)
    print()
    
    # Създаваме обект за разширени аналитични функции
    aaf = create_analytical_functions()
    
    # ===== ЧАСТ 1: АНАЛИТИЧНА ФУНКЦИЯ T(z) =====
    print("📊 ЧАСТ 1: АНАЛИТИЧНА ФУНКЦИЯ T(z)")
    print("-" * 50)
    
    # Тестваме различни стойности на z
    z_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    print("Сравнение между интегрална и приближена форма:")
    print("z\t\tT_integral\tT_approx\tГрешка (%)")
    print("-" * 60)
    
    for z in z_test:
        t_integral = aaf.analytical_t_z_integral(z)
        t_approx = aaf.analytical_t_z_approximation(z)
        error = abs(t_integral - t_approx) / t_integral * 100
        print(f"{z:.1f}\t\t{t_integral:.6f}\t{t_approx:.6f}\t{error:.2f}%")
    
    print()
    
    # ===== ЧАСТ 2: ФУНКЦИЯ a(t_abs) =====
    print("📊 ЧАСТ 2: ФУНКЦИЯ a(t_abs)")
    print("-" * 50)
    
    # Създаваме функцията за мащабния фактор
    aaf.create_scale_factor_function()
    
    # Тестваме различни времена
    t_test = np.array([1.0, 3.0, 5.0, 8.0, 10.0, 13.8])
    
    print("Еволюция на мащабния фактор:")
    print("t_abs [Gyr]\ta(t_abs)\t\tz съответно")
    print("-" * 50)
    
    for t in t_test:
        a_val = aaf._scale_factor_function(t)
        z_val = (1.0 / a_val) - 1.0
        print(f"{t:.1f}\t\t{a_val:.6f}\t\t{z_val:.3f}")
    
    print()
    
    # ===== ЧАСТ 3: ПАРАМЕТЪР НА HUBBLE H(t_abs) =====
    print("📊 ЧАСТ 3: ПАРАМЕТЪР НА HUBBLE H(t_abs)")
    print("-" * 50)
    
    print("Еволюция на параметъра на Hubble:")
    print("t_abs [Gyr]\tH(t_abs) [km/s/Mpc]")
    print("-" * 40)
    
    for t in t_test:
        H_val = aaf.hubble_parameter_abs_time(t)
        print(f"{t:.1f}\t\t{H_val:.2f}")
    
    print()
    
    # ===== ЧАСТ 4: НАТУРАЛНА МЕТРИЧНА ТРАНСФОРМАЦИЯ =====
    print("📊 ЧАСТ 4: НАТУРАЛНА МЕТРИЧНА ТРАНСФОРМАЦИЯ")
    print("-" * 50)
    
    print("Трансформация към натурално време:")
    print("t_abs [Gyr]\tτ [натурални единици]")
    print("-" * 40)
    
    for t in t_test:
        tau_val = aaf.natural_metric_transformation(t)
        print(f"{t:.1f}\t\t{tau_val:.6f}")
    
    print()
    
    # ===== ЧАСТ 5: РАЗШИРЕН АНАЛИЗ ДО z > 2 =====
    print("📊 ЧАСТ 5: РАЗШИРЕН АНАЛИЗ ДО z > 2")
    print("-" * 50)
    
    extended_results = aaf.extended_z_range_analysis(z_max=10.0)
    
    print("Ключови космологични епохи:")
    for epoch, data in extended_results['key_epochs'].items():
        print(f"\n{epoch.upper()}:")
        print(f"  z = {data['z']}")
        if data['t_abs'] is not None:
            print(f"  t_abs = {data['t_abs']:.3f} Gyr")
        print(f"  a = {data['a']:.6f}")
    
    print()
    
    # ===== ЧАСТ 6: СЪЗДАВАНЕ НА ГРАФИКИ =====
    print("📊 ЧАСТ 6: ВИЗУАЛИЗАЦИЯ")
    print("-" * 50)
    
    # Създаваме детайлни графики
    create_comprehensive_plots(aaf)
    
    # ===== ЧАСТ 7: ГЕНЕРИРАНЕ НА ПОДРОБЕН ДОКЛАД =====
    print("📊 ЧАСТ 7: ПОДРОБЕН ДОКЛАД")
    print("-" * 50)
    
    # Генерираме обобщен доклад
    report = aaf.comprehensive_analysis_report()
    
    # Записваме доклада
    with open('analysis/ADVANCED_ANALYTICAL_FUNCTIONS_REPORT.md', 'w', encoding='utf-8') as f:
        f.write("# Доклад за разширените аналитични функции\n\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n")
    
    print("✅ Докладът е записан в analysis/ADVANCED_ANALYTICAL_FUNCTIONS_REPORT.md")
    print()
    
    # ===== ЧАСТ 8: СРАВНЕНИЕ С PANTHEON+ =====
    print("📊 ЧАСТ 8: ПРОВЕРКА НА СЪВМЕСТИМОСТТА")
    print("-" * 50)
    
    # Бърз тест със z стойности от Pantheon+
    z_pantheon = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    pantheon_results = quick_t_z_analysis(z_pantheon)
    
    print("Бърз анализ с типични z стойности от Pantheon+:")
    print("z\t\tT(z)\t\tt_abs [Gyr]\ta(z)")
    print("-" * 50)
    
    for i, z in enumerate(z_pantheon):
        t_val = pantheon_results['t_integral'][i]
        t_abs_val = pantheon_results['t_absolute'][i]
        a_val = pantheon_results['scale_factor'][i]
        print(f"{z:.1f}\t\t{t_val:.6f}\t{t_abs_val:.3f}\t\t{a_val:.6f}")
    
    print()
    print("=" * 80)
    print("✅ АНАЛИЗЪТ ЗАВЪРШИ УСПЕШНО!")
    print("=" * 80)


def create_comprehensive_plots(aaf):
    """
    Създава обширни графики на всички аналитични функции
    """
    # Конфигурация за по-добри графики
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 12
    
    # Създаваме основната фигура
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ===== ГРАФИК 1: T(z) функция - интегрална vs приближена =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    z_range = np.logspace(-3, 1, 1000)
    t_integral = np.array([aaf.analytical_t_z_integral(z) for z in z_range])
    t_approx = np.array([aaf.analytical_t_z_approximation(z) for z in z_range])
    
    ax1.loglog(z_range, t_integral, 'b-', linewidth=2, label='Интегрална форма')
    ax1.loglog(z_range, t_approx, 'r--', linewidth=2, label='Приближена форма')
    ax1.set_xlabel('Червено отместване z')
    ax1.set_ylabel('T(z)')
    ax1.set_title('Аналитична функция T(z)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== ГРАФИК 2: Грешка на приближението =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    error = np.abs(t_integral - t_approx) / t_integral * 100
    ax2.semilogx(z_range, error, 'g-', linewidth=2)
    ax2.set_xlabel('Червено отместване z')
    ax2.set_ylabel('Грешка (%)')
    ax2.set_title('Грешка на приближението T(z)')
    ax2.grid(True, alpha=0.3)
    
    # ===== ГРАФИК 3: a(t_abs) функция =====
    ax3 = fig.add_subplot(gs[0, 2])
    
    if aaf._scale_factor_function is None:
        aaf.create_scale_factor_function()
    
    t_abs_range = np.linspace(0.1, 13.8, 1000)
    a_values = np.array([aaf._scale_factor_function(t) for t in t_abs_range])
    
    ax3.plot(t_abs_range, a_values, 'purple', linewidth=2)
    ax3.set_xlabel('Абсолютно време t_abs [Gyr]')
    ax3.set_ylabel('Мащабен фактор a(t_abs)')
    ax3.set_title('Зависимост a(t_abs)')
    ax3.grid(True, alpha=0.3)
    
    # Маркираме днешния ден
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Днес (a=1)')
    ax3.axvline(x=13.8, color='red', linestyle='--', alpha=0.7)
    ax3.legend()
    
    # ===== ГРАФИК 4: H(t_abs) функция =====
    ax4 = fig.add_subplot(gs[1, 0])
    
    H_values = np.array([aaf.hubble_parameter_abs_time(t) for t in t_abs_range])
    
    ax4.plot(t_abs_range, H_values, 'orange', linewidth=2)
    ax4.set_xlabel('Абсолютно време t_abs [Gyr]')
    ax4.set_ylabel('H(t_abs) [km/s/Mpc]')
    ax4.set_title('Параметър на Hubble H(t_abs)')
    ax4.grid(True, alpha=0.3)
    
    # Маркираме днешната стойност
    ax4.axhline(y=70.0, color='red', linestyle='--', alpha=0.7, label='H₀ = 70 km/s/Mpc')
    ax4.axvline(x=13.8, color='red', linestyle='--', alpha=0.7)
    ax4.legend()
    
    # ===== ГРАФИК 5: Натурална метрична трансформация =====
    ax5 = fig.add_subplot(gs[1, 1])
    
    tau_values = np.array([aaf.natural_metric_transformation(t) for t in t_abs_range])
    
    ax5.plot(t_abs_range, tau_values, 'cyan', linewidth=2)
    ax5.set_xlabel('Абсолютно време t_abs [Gyr]')
    ax5.set_ylabel('Натурално време τ')
    ax5.set_title('Натурална метрична трансформация')
    ax5.grid(True, alpha=0.3)
    
    # ===== ГРАФИК 6: Разширен z диапазон =====
    ax6 = fig.add_subplot(gs[1, 2])
    
    extended_results = aaf.extended_z_range_analysis(z_max=10.0)
    z_extended = extended_results['z_range']
    t_extended = extended_results['t_abs_values']
    
    ax6.semilogx(z_extended, t_extended, 'brown', linewidth=2)
    ax6.set_xlabel('Червено отместване z')
    ax6.set_ylabel('Абсолютно време t_abs [Gyr]')
    ax6.set_title('Разширен диапазон z > 2')
    ax6.grid(True, alpha=0.3)
    
    # Маркираме ключови епохи
    for epoch, data in extended_results['key_epochs'].items():
        if data['t_abs'] is not None:
            ax6.scatter(data['z'], data['t_abs'], s=100, label=f"{epoch}")
    
    ax6.legend()
    
    # ===== ГРАФИК 7: Съотношение da/dt =====
    ax7 = fig.add_subplot(gs[2, 0])
    
    # Пресмятаме da/dt числено
    dt = 0.01
    da_dt = np.gradient(a_values, dt)
    
    ax7.plot(t_abs_range, da_dt, 'magenta', linewidth=2)
    ax7.set_xlabel('Абсолютно време t_abs [Gyr]')
    ax7.set_ylabel('da/dt [1/Gyr]')
    ax7.set_title('Скорост на промяна на мащабния фактор')
    ax7.grid(True, alpha=0.3)
    
    # ===== ГРАФИК 8: Сравнение на метриките =====
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Нормализираме за сравнение
    t_norm = t_abs_range / 13.8
    tau_norm = tau_values / np.max(tau_values)
    
    ax8.plot(t_norm, t_norm, 'k-', linewidth=2, label='Линейно време')
    ax8.plot(t_norm, tau_norm, 'c-', linewidth=2, label='Натурално време')
    ax8.set_xlabel('Нормализирано време')
    ax8.set_ylabel('Нормализирана стойност')
    ax8.set_title('Сравнение на времевите метрики')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # ===== ГРАФИК 9: Фазов диаграм =====
    ax9 = fig.add_subplot(gs[2, 2])
    
    ax9.plot(a_values, H_values, 'red', linewidth=2)
    ax9.set_xlabel('Мащабен фактор a')
    ax9.set_ylabel('H(a) [km/s/Mpc]')
    ax9.set_title('Фазов диаграм H(a)')
    ax9.grid(True, alpha=0.3)
    
    # Маркираме днешното състояние
    ax9.scatter(1.0, 70.0, s=100, color='red', marker='*', label='Днес')
    ax9.legend()
    
    # Записваме графиката
    plt.suptitle('Обширен анализ на разширените аналитични функции', fontsize=16, y=0.98)
    plt.savefig('analysis/comprehensive_analytical_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Графиките са записани в analysis/comprehensive_analytical_plots.png")


if __name__ == "__main__":
    main() 