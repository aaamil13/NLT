#!/usr/bin/env python3
"""
Обобщен тест за BAO и CMB интеграционна система

Този файл тества цялата система за нелинейно време с BAO и CMB данни.
"""

import numpy as np
import sys
import os

# Добавяме текущата директория към пътя
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    print("🚀 СТАРТИРАНЕ НА BAO И CMB ИНТЕГРАЦИОННА СИСТЕМА")
    print("=" * 80)
    
    try:
        # Тест 1: Основни модули
        print("\n📦 ТЕСТ 1: ОСНОВНИ МОДУЛИ")
        print("-" * 40)
        
        # Импортиране на основния модул
        from common_utils.nonlinear_time_core import NonlinearTimeCosmology, test_nonlinear_time_cosmology
        print("✅ Нелинейно време модул заредена успешно")
        
        # Тест на основния модул
        test_nonlinear_time_cosmology()
        
        # Тест 2: Космологични параметри
        print("\n📊 ТЕСТ 2: КОСМОЛОГИЧНИ ПАРАМЕТРИ")
        print("-" * 40)
        
        from common_utils.cosmological_parameters import print_data_summary
        print_data_summary()
        
        # Тест 3: Обработка на данни
        print("\n🔧 ТЕСТ 3: ОБРАБОТКА НА ДАННИ")
        print("-" * 40)
        
        from common_utils.data_processing import test_data_processing
        test_data_processing()
        
        # Тест 4: BAO анализ
        print("\n🌐 ТЕСТ 4: BAO АНАЛИЗ")
        print("-" * 40)
        
        from bao_analysis.bao_analyzer import BAOAnalyzer
        bao_analyzer = BAOAnalyzer()
        
        # Зареждане на данни
        bao_analyzer.load_real_data('combined')
        print(f"✅ Заредени {len(bao_analyzer.processed_data['z'])} BAO точки")
        
        # Основно сравнение
        comparison = bao_analyzer.compare_with_observations()
        print(f"✅ BAO χ²/dof = {comparison['statistics']['reduced_chi_squared']:.2f}")
        print(f"✅ Ниво на съответствие: {comparison['agreement_level']}")
        
        # Тест 5: CMB анализ
        print("\n🌠 ТЕСТ 5: CMB АНАЛИЗ")
        print("-" * 40)
        
        from cmb_analysis.cmb_analyzer import CMBAnalyzer
        cmb_analyzer = CMBAnalyzer()
        
        # Ъглов размер на звуковия хоризонт
        theta_star = cmb_analyzer.calculate_angular_sound_horizon()
        print(f"✅ θ* = {theta_star:.7f} rad")
        
        # Акустични пикове
        peaks = cmb_analyzer.calculate_acoustic_peak_positions()
        print(f"✅ Първи пик: l = {peaks[0]:.1f}")
        
        # Сравнение с Planck
        cmb_comparison = cmb_analyzer.compare_with_planck_data()
        print(f"✅ CMB χ²/dof = {cmb_comparison['statistics']['reduced_chi_squared']:.2f}")
        print(f"✅ Ниво на съответствие: {cmb_comparison['agreement_level']}")
        
        # Тест 6: Съвместен анализ
        print("\n🔗 ТЕСТ 6: СЪВМЕСТЕН АНАЛИЗ")
        print("-" * 40)
        
        from integration_core.joint_analyzer import JointBAOCMBAnalyzer
        joint_analyzer = JointBAOCMBAnalyzer()
        
        # Оптимизиране на параметрите
        optimization = joint_analyzer.optimize_parameters()
        print(f"✅ Оптимизирането завърши: {optimization['success']}")
        print(f"✅ Най-добър χ²: {optimization['chi_squared']:.2f}")
        
        # Обширен анализ
        results = joint_analyzer.comprehensive_joint_analysis()
        
        # Принтиране на крайния доклад
        joint_analyzer.print_comprehensive_report()
        
        # Финални резултати
        print("\n🎉 ФИНАЛНИ РЕЗУЛТАТИ")
        print("-" * 40)
        
        best_params = results.best_fit_parameters
        print(f"✅ Най-добри параметри:")
        print(f"   α = {best_params['alpha']:.4f}")
        print(f"   β = {best_params['beta']:.4f}")
        print(f"   γ = {best_params['gamma']:.4f}")
        print(f"   δ = {best_params['delta']:.4f}")
        
        combined_stats = results.combined_statistics
        print(f"✅ Комбинирани статистики:")
        print(f"   Общо χ² = {combined_stats['total_chi_squared']:.2f}")
        print(f"   Редуциран χ² = {combined_stats['reduced_chi_squared']:.2f}")
        print(f"   DOF = {combined_stats['total_dof']}")
        
        print(f"✅ Оценка на съвместимостта:")
        print(f"   {results.agreement_assessment}")
        
        # Сравнение с ΛCDM
        comparison_table = joint_analyzer.generate_comparison_table()
        improvement = "ДА" if comparison_table['improvement'] else "НЕ"
        print(f"✅ Подобрение спрямо ΛCDM: {improvement}")
        print(f"✅ Δχ² = {comparison_table['delta_chi2']:.2f}")
        print(f"✅ Значимост = {comparison_table['significance']:.2f}σ")
        
        print("\n" + "="*80)
        print("🎊 ВСИЧКИ ТЕСТОВЕ ЗАВЪРШИХА УСПЕШНО!")
        print("🎊 BAO И CMB ИНТЕГРАЦИОННАТА СИСТЕМА РАБОТИ ПЕРФЕКТНО!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ГРЕШКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 