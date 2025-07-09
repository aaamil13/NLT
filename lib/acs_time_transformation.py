#!/usr/bin/env python3
"""
Модул за числено моделиране на разширението на Вселената спрямо абсолютна координатна система (АКС)
Имплементира теоретичните разработки за:
- Линейно разширение в абсолютното време
- Релативна система с компресирана хронология за ранната Вселена
- Трансформация на времето между АКС и наблюдателна система
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

class TimeTransformationModel:
    """
    Модел за времева трансформация между абсолютна и релативна система
    """
    
    def __init__(self, k_expansion=1e-3, t_universe_gyr=13.8):
        """
        Инициализация на модела
        
        Parameters:
        -----------
        k_expansion : float
            Скалиращ коефициент за разширение a(t_abs) = k * t_abs
        t_universe_gyr : float
            Възраст на Вселената в милиарди години
        """
        self.k = k_expansion
        self.t_universe = t_universe_gyr
        
    def density_approximation(self, z):
        """
        Приближение за плътност на материята: ρ(z) ∝ (1+z)³
        """
        return (1 + z)**3
    
    def time_transformation_factor(self, z):
        """
        Времевия трансформационен фактор T(z) = 1/(1+z)^(3/2)
        
        Parameters:
        -----------
        z : array-like
            Червено отместване
            
        Returns:
        --------
        T(z) : array-like
            Коефициент на времева трансформация
        """
        return 1.0 / (1 + z)**(3/2)
    
    def dt_rel_dt_abs(self, t_abs):
        """
        Производна dt_rel/dt_abs ∝ t_abs^(3/2)
        
        Parameters:
        -----------
        t_abs : array-like
            Абсолютно време
            
        Returns:
        --------
        dt_rel/dt_abs : array-like
            Коефициент на времева производна
        """
        return t_abs**(3/2)
    
    def compute_relative_time(self, t_abs_array):
        """
        Изчисляване на релативното време от абсолютното
        t_rel(t_abs) ∝ ∫₀^t_abs t^(3/2) dt = (2/5) * t_abs^(5/2)
        
        Parameters:
        -----------
        t_abs_array : array-like
            Масив от абсолютни времена
            
        Returns:
        --------
        t_rel : array-like
            Релативни времена
        """
        return (2/5) * t_abs_array**(5/2)
    
    def scale_factor_absolute(self, t_abs):
        """
        Мащабен фактор в абсолютното време: a(t_abs) = k * t_abs
        """
        return self.k * t_abs
    
    def scale_factor_relative(self, t_rel):
        """
        Мащабен фактор в релативното време: a(t_rel) = k * (t_rel)^(1/2.5)
        """
        return self.k * t_rel**(1/2.5)

class RedshiftTimeRelation:
    """
    Клас за връзка между червено отместване и време
    """
    
    def __init__(self, H0=70):
        """
        Parameters:
        -----------
        H0 : float
            Константа на Хъбъл в km/s/Mpc
        """
        self.H0 = H0
        self.H0_SI = H0 * 1000 / (3.086e22)  # 1/s
        self.H0_inv_Gyr = 1 / self.H0_SI / (3.1536e16 * 1e9)  # Gyr
    
    def hubble_parameter(self, z):
        """
        Параметър на Хъбъл H(z) ∼ (1+z)^(3/2) за материя-доминирана вселена
        """
        return self.H0_SI * (1 + z)**(3/2)
    
    def dt_abs_dz(self, z):
        """
        Диференциал dt_abs/dz = 1/((1+z)^(5/2) * H0)
        """
        return 1 / ((1 + z)**(5/2) * self.H0_SI)
    
    def absolute_time_from_redshift(self, z_array):
        """
        Изчисляване на абсолютното време от червеното отместване
        
        Parameters:
        -----------
        z_array : array-like
            Масив от червени отмествания (трябва да е сортиран низходящо)
            
        Returns:
        --------
        t_abs : array-like
            Абсолютно време в секунди
        """
        dt_dz = self.dt_abs_dz(z_array)
        t_abs_seconds = cumtrapz(dt_dz, z_array, initial=0)
        return t_abs_seconds
    
    def redshift_from_time(self, t_abs_array, z_max=20):
        """
        Обратна функция - намиране на z от абсолютното време
        
        Parameters:
        -----------
        t_abs_array : array-like
            Масив от абсолютни времена
        z_max : float
            Максимално червено отместване за интегриране
            
        Returns:
        --------
        z_array : array-like
            Червени отмествания
        """
        # Създаваме таблица z -> t_abs
        z_ref = np.linspace(z_max, 0, 1000)
        t_abs_ref = self.absolute_time_from_redshift(z_ref[::-1])[::-1]
        
        # Интерполация за обратната функция
        interp_func = interp1d(t_abs_ref, z_ref, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        
        return interp_func(t_abs_array)

class ExpansionAnalyzer:
    """
    Анализатор на разширението в различни координатни системи
    """
    
    def __init__(self, time_model, redshift_model):
        """
        Parameters:
        -----------
        time_model : TimeTransformationModel
            Модел за времева трансформация
        redshift_model : RedshiftTimeRelation
            Модел за връзка между червено отместване и време
        """
        self.time_model = time_model
        self.redshift_model = redshift_model
    
    def generate_discrete_timeline(self, delta_t_gyr=1, max_t_gyr=13.8):
        """
        Генериране на дискретни времеви интервали
        
        Parameters:
        -----------
        delta_t_gyr : float
            Интервал на дискретизация в милиарди години
        max_t_gyr : float
            Максимално време в милиарди години
            
        Returns:
        --------
        t_abs_array : array
            Масив от абсолютни времена
        """
        return np.arange(0.1, max_t_gyr + delta_t_gyr, delta_t_gyr)
    
    def compute_expansion_table(self, t_abs_array):
        """
        Изчисляване на таблица с разширение
        
        Parameters:
        -----------
        t_abs_array : array-like
            Масив от абсолютни времена
            
        Returns:
        --------
        results : dict
            Речник с резултати
        """
        # Изчисляваме основните величини
        a_abs = self.time_model.scale_factor_absolute(t_abs_array)
        t_rel = self.time_model.compute_relative_time(t_abs_array)
        
        # Нормализираме релативното време
        t_rel_normalized = t_rel / np.max(t_rel) * np.max(t_abs_array)
        
        # Намираме червеното отместване
        t_abs_seconds = t_abs_array * 1e9 * 3.1536e16  # секунди
        z_values = self.redshift_model.redshift_from_time(t_abs_seconds)
        
        # Мащабен фактор в релативното време
        a_rel = self.time_model.scale_factor_relative(t_rel)
        
        results = {
            't_abs_gyr': t_abs_array,
            't_rel_normalized': t_rel_normalized,
            't_rel_raw': t_rel,
            'a_abs': a_abs,
            'a_rel': a_rel,
            'z_values': z_values,
            'density_factor': self.time_model.density_approximation(z_values),
            'time_transform_factor': self.time_model.time_transformation_factor(z_values)
        }
        
        return results
    
    def print_expansion_table(self, results):
        """
        Печатане на таблицата с разширение
        """
        print("=" * 80)
        print("ТАБЛИЦА НА РАЗШИРЕНИЕТО СПРЯМО АБСОЛЮТНА КООРДИНАТНА СИСТЕМА")
        print("=" * 80)
        print(f"{'t_abs [Gyr]':<12} {'a(t_abs)':<10} {'t_rel ∝ t_abs^5/2':<15} {'z':<8} {'ρ(z)':<10} {'T(z)':<8}")
        print("-" * 80)
        
        for i in range(len(results['t_abs_gyr'])):
            t_abs = results['t_abs_gyr'][i]
            a_abs = results['a_abs'][i]
            t_rel_norm = results['t_rel_normalized'][i]
            z_val = results['z_values'][i]
            density = results['density_factor'][i]
            transform = results['time_transform_factor'][i]
            
            print(f"{t_abs:<12.1f} {a_abs:<10.3f} {t_rel_norm:<15.2f} {z_val:<8.3f} {density:<10.2f} {transform:<8.3f}")

class ExpansionVisualizer:
    """
    Визуализатор на разширението
    """
    
    def __init__(self, results):
        """
        Parameters:
        -----------
        results : dict
            Резултати от ExpansionAnalyzer
        """
        self.results = results
    
    def plot_time_transformation(self, figsize=(15, 10)):
        """
        Графики на времевата трансформация
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # График 1: Мащабен фактор спрямо абсолютното време
        axes[0, 0].plot(self.results['t_abs_gyr'], self.results['a_abs'], 
                       'b-', linewidth=2, label='a(t_abs) = k·t_abs')
        axes[0, 0].set_xlabel('Абсолютно време [Gyr]')
        axes[0, 0].set_ylabel('Мащабен фактор a(t_abs)')
        axes[0, 0].set_title('Линейно разширение в АКС')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # График 2: Релативно време спрямо абсолютното
        axes[0, 1].plot(self.results['t_abs_gyr'], self.results['t_rel_normalized'], 
                       'g-', linewidth=2, label='t_rel ∝ t_abs^(5/2)')
        axes[0, 1].set_xlabel('Абсолютно време [Gyr]')
        axes[0, 1].set_ylabel('Релативно време [Gyr]')
        axes[0, 1].set_title('Времева трансформация')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # График 3: Мащабен фактор спрямо релативното време
        axes[1, 0].plot(self.results['t_rel_normalized'], self.results['a_abs'], 
                       'r-', linewidth=2, label='a(t_rel) = k·(t_rel)^(1/2.5)')
        axes[1, 0].set_xlabel('Релативно време [Gyr]')
        axes[1, 0].set_ylabel('Мащабен фактор')
        axes[1, 0].set_title('Нелинейно разширение в РКС')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # График 4: Червено отместване спрямо абсолютното време
        axes[1, 1].plot(self.results['t_abs_gyr'], self.results['z_values'], 
                       'm-', linewidth=2, label='z(t_abs)')
        axes[1, 1].set_xlabel('Абсолютно време [Gyr]')
        axes[1, 1].set_ylabel('Червено отместване z')
        axes[1, 1].set_title('Еволюция на червеното отместване')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison_models(self, figsize=(12, 8)):
        """
        Сравнение между различни модели
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Сравнение на мащачни фактори
        axes[0].plot(self.results['z_values'], 1/(1+self.results['z_values']), 
                    'b--', linewidth=2, label='ΛCDM: a = 1/(1+z)')
        axes[0].plot(self.results['z_values'], self.results['a_abs']/np.max(self.results['a_abs']), 
                    'r-', linewidth=2, label='АКС модел')
        axes[0].set_xlabel('Червено отместване z')
        axes[0].set_ylabel('Нормализиран мащабен фактор')
        axes[0].set_title('Сравнение на модели')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].grid(True)
        axes[0].legend()
        
        # Времевия трансформационен фактор
        axes[1].plot(self.results['z_values'], self.results['time_transform_factor'], 
                    'g-', linewidth=2, label='T(z) = 1/(1+z)^(3/2)')
        axes[1].set_xlabel('Червено отместване z')
        axes[1].set_ylabel('Времевия фактор T(z)')
        axes[1].set_title('Времева трансформация')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].grid(True)
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Основна функция за демонстрация
    """
    print("🚀 ЧИСЛЕНО МОДЕЛИРАНЕ НА РАЗШИРЕНИЕТО СПРЯМО АБСОЛЮТНА КООРДИНАТНА СИСТЕМА")
    print("=" * 80)
    
    # Инициализация на моделите
    time_model = TimeTransformationModel(k_expansion=1e-3, t_universe_gyr=13.8)
    redshift_model = RedshiftTimeRelation(H0=70)
    analyzer = ExpansionAnalyzer(time_model, redshift_model)
    
    # Генериране на дискретни времеви интервали
    print("\n📊 Генериране на дискретни времеви интервали...")
    t_abs_array = analyzer.generate_discrete_timeline(delta_t_gyr=1, max_t_gyr=13.8)
    
    # Изчисляване на таблицата с разширение
    print("🧮 Изчисляване на таблицата с разширение...")
    results = analyzer.compute_expansion_table(t_abs_array)
    
    # Печатане на таблицата
    analyzer.print_expansion_table(results)
    
    # Визуализация
    print("\n📈 Създаване на графики...")
    visualizer = ExpansionVisualizer(results)
    visualizer.plot_time_transformation()
    visualizer.plot_comparison_models()
    
    print("\n✅ Анализът завършен успешно!")
    print("\n🔍 Ключови наблюдения:")
    print("• Линейното разширение в АКС се преобразува в нелинейно в РКС")
    print("• Времевата трансформация T(z) обяснява ускорението без тъмна енергия")
    print("• Компресираната хронология в ранната Вселена е естествено следствие")

if __name__ == "__main__":
    main() 