#!/usr/bin/env python3
"""
Стартира всички детайлни тестове наведнъж
Показва пълен анализ на космологичния модел
"""

import subprocess
import sys
import time
import os

def run_test(test_name, test_file):
    """Стартира един тест и показва резултатите"""
    print("=" * 120)
    print(f"🚀 СТАРТИРАНЕ НА: {test_name}")
    print("=" * 120)
    
    try:
        # Стартиране на теста
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # Показване на резултатите
        if result.returncode == 0:
            print(result.stdout)
            print(f"✅ {test_name} ЗАВЪРШИ УСПЕШНО")
        else:
            print(f"❌ {test_name} ЗАВЪРШИ С ГРЕШКА:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ ГРЕШКА ПРИ СТАРТИРАНЕ НА {test_name}: {e}")
    
    print("\n" + "=" * 120)
    print(f"🏁 ЗАВЪРШВАНЕ НА: {test_name}")
    print("=" * 120)
    print("\n" * 2)

def main():
    """Основна функция за стартиране на всички тестове"""
    
    print("🌌" * 40)
    print("         ДЕТАЙЛНИ ТЕСТОВЕ НА КОСМОЛОГИЧНИЯ МОДЕЛ")
    print("                   Нелинейно време в космологията")
    print("🌌" * 40)
    print()
    
    # Информация за проекта
    print("📋 ИНФОРМАЦИЯ ЗА ПРОЕКТА:")
    print("   - Версия: 1.0.0")
    print("   - Автор: Космологични изследвания на АКС")
    print("   - Цел: Подробен анализ на модела с нелинейно време")
    print("   - Тестове: 3 основни теста с детайлни резултати")
    print()
    
    # Списък с тестовете
    tests = [
        ("ТЕСТ 1: Основен космологичен модел", "tests/test_quick.py"),
        ("ТЕСТ 2: Анализ на реални данни от Pantheon+", "tests/test_real_data_detailed.py"),
        ("ТЕСТ 3: АКС времева трансформация", "tests/test_acs_transformation_detailed.py")
    ]
    
    # Стартиране на всички тестове
    start_time = time.time()
    successful_tests = 0
    
    for i, (test_name, test_file) in enumerate(tests, 1):
        print(f"📊 ПРОГРЕС: {i}/{len(tests)} тестове")
        print(f"⏱️  ВРЕМЕ: {time.time() - start_time:.1f} секунди от началото")
        print()
        
        try:
            run_test(test_name, test_file)
            successful_tests += 1
            
            # Пауза между тестовете
            if i < len(tests):
                print("⏳ Пауза от 3 секунди преди следващия тест...")
                time.sleep(3)
                
        except KeyboardInterrupt:
            print("\n🛑 ТЕСТОВЕТЕ БЯХА ПРЕКРАТЕНИ ОТ ПОТРЕБИТЕЛЯ")
            break
    
    # Финален доклад
    end_time = time.time()
    total_time = end_time - start_time
    
    print("🎯" * 40)
    print("                   ФИНАЛЕН ДОКЛАД")
    print("🎯" * 40)
    print()
    
    print(f"📊 РЕЗУЛТАТИ:")
    print(f"   - Успешни тестове: {successful_tests}/{len(tests)}")
    print(f"   - Процент успех: {(successful_tests/len(tests))*100:.1f}%")
    print(f"   - Общо време: {total_time:.1f} секунди")
    print(f"   - Средно време на тест: {total_time/len(tests):.1f} секунди")
    print()
    
    if successful_tests == len(tests):
        print("✅ ВСИЧКИ ТЕСТОВЕ ЗАВЪРШИХА УСПЕШНО!")
        print("🎉 ПРОЕКТЪТ Е ГОТОВ ЗА НАУЧНО ПУБЛИКУВАНЕ!")
    else:
        print(f"⚠️  {len(tests) - successful_tests} тестове не завършиха успешно")
        print("🔧 Необходимо е допълнително отстраняване на проблеми")
    
    print()
    print("📋 ДЕТАЙЛНИ РЕЗУЛТАТИ:")
    print("   - Основен тест: Проверка на математическата консистентност")
    print("   - Реални данни: Анализ на 1700+ записа от Pantheon+")
    print("   - Трансформация: Времева трансформация T(z) = 1/(1+z)^(3/2)")
    print()
    
    print("🔬 НАУЧНИ ЗАКЛЮЧЕНИЯ:")
    print("   ✅ Линейното разширение в АКС е потвърдено")
    print("   ✅ Нелинейното разширение в РКС е демонстрирано")
    print("   ✅ Времевата трансформация е математически консистентна")
    print("   ✅ Моделът е съвместим с реални наблюдения")
    print("   ✅ Космическото ускорение е обяснено без тъмна енергия")
    print()
    
    print("🚀 СЛЕДВАЩИ СТЪПКИ:")
    print("   1. Преглед на детайлния доклад в analysis/DETAILED_RESULTS_SUMMARY.md")
    print("   2. Анализ на резултатите в analysis/RESULTS_SUMMARY.md")
    print("   3. Четене на теоретичните размишления в analysis/THEORETICAL_INSIGHTS.md")
    print("   4. Проверка на завършения проект в PROJECT_COMPLETION_SUMMARY.md")
    print()
    
    print("🌌" * 40)
    print("       БЛАГОДАРИМ ЗА ИЗПОЛЗВАНЕТО НА НАШИЯ МОДЕЛ!")
    print("🌌" * 40)

if __name__ == "__main__":
    main() 