#!/usr/bin/env python3
"""
Анализ на реални данни за АКС (Абсолютна Координатна Система)
Използва данни от Pantheon+ за намиране на единна АКС базирана на 13.8 млрд. години
и предполага линейно разширение за търсене на други АКС с равни интервали.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, curve_fit
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class PantheonDataLoader:
    """Зарежда и обработва данни от Pantheon+"""
    
    def __init__(self, data_path=r"D:\MyPRJ\Python\NotLinearTime\test_2\data\Pantheon+_Data\4_DISTANCES_AND_COVAR\Pantheon+SH0ES.dat"):
        self.data_path = data_path
        self.data = None
        self.age_universe = 13.8e9  # години
        
    def load_data(self):
        """Зарежда данните от Pantheon+"""
        try:
            self.data = pd.read_csv(self.data_path, sep='\s+', comment='#')
            print(f"Заредени {len(self.data)} записа от Pantheon+")
            return True
        except Exception as e:
            print(f"Грешка при зареждане на данните: {e}")
            return False
    
    def get_redshift_data(self, max_redshift=2.0):
        """Извлича данни за червено отместване и разстояние"""
        if self.data is None:
            return None, None
            
        # Филтрираме данни с валидни стойности
        valid_data = self.data[
            (self.data['zHD'] > 0) & 
            (self.data['zHD'] < max_redshift) & 
            (self.data['MU_SH0ES'] > 0) & 
            (self.data['MU_SH0ES'] < 50)
        ].copy()
        
        # Сортираме по червено отместване
        valid_data = valid_data.sort_values('zHD')
        
        redshift = valid_data['zHD'].values
        distance_modulus = valid_data['MU_SH0ES'].values
        
        print(f"Извлечени {len(redshift)} валидни записа за анализ")
        return redshift, distance_modulus

class UnifiedACSFinder:
    """Намира единна АКС базирана на възрастта на Вселената"""
    
    def __init__(self, age_universe=13.8e9):
        self.age_universe = age_universe  # години
        self.H0 = 70.0  # km/s/Mpc - Хъбълова константа
        self.c = 2.998e5  # km/s - скорост на светлината
        self.unified_acs = None
        
    def redshift_to_age(self, z):
        """Преобразува червено отместване в възраст на Вселената"""
        # Опростена формула за космологично време
        # t = (2/3) * (1/H0) * (1/(1+z))^(3/2) за материя-доминирана Вселена
        # По-точна формула включваща тъмна енергия
        return (2.0/3.0) * (1.0/self.H0) * (1.0/(1.0 + z))**(1.5) * 9.777e11  # години
    
    def age_to_redshift(self, age):
        """Преобразува възраст в червено отместване"""
        # Обратна формула
        h_factor = age / (9.777e11 * 2.0/3.0 * 1.0/self.H0)
        if h_factor > 0:
            return (1.0/h_factor)**(2.0/3.0) - 1.0
        else:
            return 0.0
    
    def find_unified_acs(self, redshift_data, distance_data):
        """Намира единна АКС за текущия момент (13.8 млрд. години)"""
        
        # Преобразуваме червено отместване в възраст
        ages = np.array([self.redshift_to_age(z) for z in redshift_data])
        
        # Филтрираме данни за възрасти по-малки от възрастта на Вселената
        valid_indices = ages < self.age_universe
        ages_valid = ages[valid_indices]
        redshift_valid = redshift_data[valid_indices]
        distance_valid = distance_data[valid_indices]
        
        print(f"Валидни данни за възрасти: {len(ages_valid)} записа")
        
        # Намираме единна АКС като екстраполираме към сегашния момент
        # Използваме линейна интерполация в log-space
        if len(ages_valid) > 10:
            # Сортираме по възраст
            sort_indices = np.argsort(ages_valid)
            ages_sorted = ages_valid[sort_indices]
            redshift_sorted = redshift_valid[sort_indices]
            
            # Интерполираме редшифта като функция на възрастта
            interp_func = interp1d(ages_sorted, redshift_sorted, 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
            
            # Единната АКС е в сегашния момент (z = 0)
            current_redshift = 0.0
            
            self.unified_acs = {
                'age': self.age_universe,
                'redshift': current_redshift,
                'time_coordinate': self.age_universe,
                'interpolation_function': interp_func
            }
            
            print(f"Единна АКС установена за възраст {self.age_universe/1e9:.1f} млрд. години")
            return self.unified_acs
        else:
            print("Недостатъчно данни за установяване на единна АКС")
            return None

class LinearACSGenerator:
    """Генерира АКС с линейни интервали от единната АКС"""
    
    def __init__(self, unified_acs):
        self.unified_acs = unified_acs
        self.linear_acs_systems = []
        
    def generate_linear_intervals(self, num_intervals=5, interval_size=2.0e9):
        """Генерира АКС с равни линейни интервали"""
        
        if self.unified_acs is None:
            print("Няма установена единна АКС")
            return []
        
        base_age = self.unified_acs['age']
        
        # Създаваме линейни интервали назад във времето
        linear_ages = []
        for i in range(num_intervals):
            age = base_age - (i * interval_size)
            if age > 0:
                linear_ages.append(age)
        
        # Създаваме АКС системи за всяка възраст
        for age in linear_ages:
            # Изчисляваме червено отместване за тази възраст
            redshift = self.age_to_redshift(age)
            
            # Създаваме АКС система
            acs_system = {
                'age': age,
                'redshift': redshift,
                'time_coordinate': age,
                'interval_from_base': base_age - age,
                'expansion_factor': base_age / age if age > 0 else 1.0
            }
            
            self.linear_acs_systems.append(acs_system)
        
        print(f"Генерирани {len(self.linear_acs_systems)} АКС системи с линейни интервали")
        return self.linear_acs_systems
    
    def age_to_redshift(self, age):
        """Преобразува възраст в червено отместване"""
        H0 = 70.0
        return (2.0/3.0 * 1.0/H0 * 9.777e11 / age)**(2.0/3.0) - 1.0

class LinearExpansionAnalyzer:
    """Анализира линейното разширение в АКС"""
    
    def __init__(self, unified_acs, linear_acs_systems):
        self.unified_acs = unified_acs
        self.linear_acs_systems = linear_acs_systems
        
    def calculate_expansion_coefficients(self):
        """Изчислява коефициенти на разширение между АКС системите"""
        
        coefficients = []
        
        for i, acs in enumerate(self.linear_acs_systems):
            if i == 0:
                # Първата АКС е базата
                coeff = 1.0
            else:
                # Коефициент на разширение спрямо базата
                base_age = self.unified_acs['age']
                current_age = acs['age']
                
                # Линейно разширение: a(t) = k*t
                coeff = base_age / current_age if current_age > 0 else 1.0
            
            coefficients.append({
                'acs_index': i,
                'age': acs['age'],
                'redshift': acs['redshift'],
                'expansion_coefficient': coeff,
                'linear_expansion': coeff
            })
        
        return coefficients
    
    def compare_with_observations(self, redshift_data, distance_data):
        """Сравнява линейното разширение с наблюденията"""
        
        # Изчисляваме теоретичните стойности за линейно разширение
        theoretical_data = []
        
        for acs in self.linear_acs_systems:
            z = acs['redshift']
            age = acs['age']
            
            # Теоретично разстояние за линейно разширение
            # d = c * z / H0 за малки z
            # За големи z използваме модификация
            if z > 0:
                theoretical_distance = self.calculate_theoretical_distance(z)
            else:
                theoretical_distance = 0.0
            
            theoretical_data.append({
                'redshift': z,
                'age': age,
                'theoretical_distance': theoretical_distance
            })
        
        return theoretical_data
    
    def calculate_theoretical_distance(self, z):
        """Изчислява теоретично разстояние за дадено червено отместване"""
        c = 2.998e5  # km/s
        H0 = 70.0    # km/s/Mpc
        
        # Опростена формула за линейно разширение
        # μ = 5 * log10(d) + 25, където d е в Mpc
        d_mpc = c * z / H0
        
        # Модул на разстоянието
        if d_mpc > 0:
            mu = 5.0 * np.log10(d_mpc) + 25.0
        else:
            mu = 0.0
        
        return mu

class RealDataACSVisualizer:
    """Визуализира резултатите от анализа на реални данни"""
    
    def __init__(self):
        self.fig_size = (12, 8)
        
    def plot_unified_acs_analysis(self, unified_acs, linear_acs_systems, 
                                 redshift_data, distance_data):
        """Рисува анализа на единната АКС и линейните интервали"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Анализ на единна АКС с реални данни от Pantheon+', fontsize=16)
        
        # График 1: Данни от Pantheon+ - редшифт vs разстояние
        ax1 = axes[0, 0]
        ax1.scatter(redshift_data, distance_data, alpha=0.6, s=30, 
                   label='Данни от Pantheon+')
        ax1.set_xlabel('Червено отместване (z)')
        ax1.set_ylabel('Модул на разстоянието (μ)')
        ax1.set_title('Наблюдения от Pantheon+')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: АКС възрасти във времето
        ax2 = axes[0, 1]
        ages = [acs['age']/1e9 for acs in linear_acs_systems]
        redshifts = [acs['redshift'] for acs in linear_acs_systems]
        
        ax2.plot(ages, redshifts, 'ro-', linewidth=2, markersize=8,
                label='АКС системи')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5,
                   label='Единна АКС (z=0)')
        ax2.set_xlabel('Възраст на Вселената (млрд. години)')
        ax2.set_ylabel('Червено отместване')
        ax2.set_title('АКС системи в линейни интервали')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График 3: Коефициенти на разширение
        ax3 = axes[1, 0]
        expansion_coeffs = [acs['expansion_factor'] for acs in linear_acs_systems]
        
        ax3.plot(ages, expansion_coeffs, 'bo-', linewidth=2, markersize=8,
                label='Коефициент на разширение')
        ax3.set_xlabel('Възраст на Вселената (млрд. години)')
        ax3.set_ylabel('Коефициент на разширение')
        ax3.set_title('Линейно разширение в АКС')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # График 4: Сравнение теория vs наблюдения
        ax4 = axes[1, 1]
        
        # Наблюдения
        ax4.scatter(redshift_data, distance_data, alpha=0.6, s=30, 
                   color='blue', label='Наблюдения')
        
        # Теоретични стойности за линейно разширение
        analyzer = LinearExpansionAnalyzer(unified_acs, linear_acs_systems)
        theoretical_data = analyzer.compare_with_observations(redshift_data, distance_data)
        
        theory_z = [data['redshift'] for data in theoretical_data if data['redshift'] > 0]
        theory_mu = [data['theoretical_distance'] for data in theoretical_data if data['redshift'] > 0]
        
        if theory_z:
            ax4.plot(theory_z, theory_mu, 'ro-', linewidth=2, markersize=8,
                    label='Теория (линейно разширение)')
        
        ax4.set_xlabel('Червено отместване (z)')
        ax4.set_ylabel('Модул на разстоянието (μ)')
        ax4.set_title('Сравнение: Теория vs Наблюдения')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_acs_timeline(self, unified_acs, linear_acs_systems):
        """Рисува времевата линия на АКС системите"""
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Данни за визуализация
        ages = [acs['age']/1e9 for acs in linear_acs_systems]
        redshifts = [acs['redshift'] for acs in linear_acs_systems]
        expansions = [acs['expansion_factor'] for acs in linear_acs_systems]
        
        # Двойна y-оста
        ax2 = ax.twinx()
        
        # График на червеното отместване
        line1 = ax.plot(ages, redshifts, 'bo-', linewidth=2, markersize=8,
                       label='Червено отместване')
        ax.set_xlabel('Възраст на Вселената (млрд. години)')
        ax.set_ylabel('Червено отместване (z)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # График на коефициента на разширение
        line2 = ax2.plot(ages, expansions, 'ro-', linewidth=2, markersize=8,
                        label='Коефициент на разширение')
        ax2.set_ylabel('Коефициент на разширение', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Маркираме единната АКС
        ax.axvline(x=unified_acs['age']/1e9, color='green', linestyle='--',
                  linewidth=2, label='Единна АКС (13.8 млрд. г.)')
        
        # Заглавие и легенда
        ax.set_title('Времева линия на АКС системи с линейни интервали', fontsize=14)
        
        # Комбинираме легендите
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def main():
    """Основна функция за анализ на реални данни"""
    
    print("=== Анализ на реални данни за АКС с Pantheon+ ===")
    
    # 1. Зареждаме данните
    loader = PantheonDataLoader()
    if not loader.load_data():
        print("Неуспешно зареждане на данните")
        return
    
    # 2. Извличаме данни за червено отместване
    redshift_data, distance_data = loader.get_redshift_data()
    if redshift_data is None:
        print("Неуспешно извличане на данни")
        return
    
    # 3. Намираме единна АКС
    acs_finder = UnifiedACSFinder()
    unified_acs = acs_finder.find_unified_acs(redshift_data, distance_data)
    if unified_acs is None:
        print("Неуспешно намиране на единна АКС")
        return
    
    # 4. Генерираме АКС с линейни интервали
    linear_generator = LinearACSGenerator(unified_acs)
    linear_acs_systems = linear_generator.generate_linear_intervals(
        num_intervals=6, interval_size=2.0e9
    )
    
    # 5. Анализираме линейното разширение
    analyzer = LinearExpansionAnalyzer(unified_acs, linear_acs_systems)
    expansion_coeffs = analyzer.calculate_expansion_coefficients()
    
    # 6. Показваме резултатите
    print("\n=== Резултати ===")
    print(f"Единна АКС: възраст = {unified_acs['age']/1e9:.1f} млрд. години")
    print(f"Генерирани {len(linear_acs_systems)} АКС системи")
    
    print("\nАКС системи с линейни интервали:")
    for i, acs in enumerate(linear_acs_systems):
        print(f"  АКС {i+1}: възраст = {acs['age']/1e9:.1f} млрд. г., "
              f"z = {acs['redshift']:.3f}, коефициент = {acs['expansion_factor']:.3f}")
    
    # 7. Визуализираме резултатите
    visualizer = RealDataACSVisualizer()
    visualizer.plot_unified_acs_analysis(unified_acs, linear_acs_systems,
                                        redshift_data, distance_data)
    visualizer.plot_acs_timeline(unified_acs, linear_acs_systems)
    
    print("\n=== Анализ завършен ===")

if __name__ == "__main__":
    main() 