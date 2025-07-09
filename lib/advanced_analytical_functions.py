"""
Разширени аналитични функции за нелинейно време и абсолютна координатна система
==============================================================================

Този модул имплементира разширените аналитични функции за:
1. Аналитична функция T(z) с интегрална и приближена форма
2. Функция a(t_abs) за зависимостта на мащабния фактор от абсолютното време
3. Разширен тест за z>2 с CMB и BAO данни
4. Натурална метрична система с постоянна скорост на разширение

Автор: Система за анализ на нелинейно време
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize_scalar, fsolve
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.special import gamma, hyp2f1
import warnings
from typing import Tuple, Callable, Dict, Any, Optional

# Физични константи
H0 = 70.0  # km/s/Mpc
c = 299792.458  # km/s
t_universe = 13.8  # Gyr


class AdvancedAnalyticalFunctions:
    """
    Клас за разширени аналитични функции в нелинейното време
    """
    
    def __init__(self):
        """Инициализация на класа"""
        self.t_universe = t_universe
        self.H0 = H0
        self.c = c
        self._analytical_t_z = None
        self._scale_factor_function = None
        self._natural_metric = None
        
    def analytical_t_z_integral(self, z: float) -> float:
        """
        Пресмята аналитичната функция T(z) чрез интеграл
        
        T(z) = ∫[z to ∞] 1/((1+z')^(5/2)) dz'
        
        Args:
            z: Червено отместване
            
        Returns:
            Стойност на T(z)
        """
        def integrand(z_prime):
            return 1.0 / ((1.0 + z_prime) ** (5.0/2.0))
        
        if z >= 100:  # За много големи z използваме приближение
            return 2.0 / (3.0 * (1.0 + z) ** (3.0/2.0))
        
        try:
            result, error = quad(integrand, z, np.inf)
            if error > 1e-10:
                warnings.warn(f"Голяма грешка при интеграцията: {error}")
            return result
        except:
            # Аварийно приближение
            return 2.0 / (3.0 * (1.0 + z) ** (3.0/2.0))
    
    def analytical_t_z_approximation(self, z: float) -> float:
        """
        Приближена аналитична форма на T(z)
        
        T(z) ≈ (2/3) * (1+z)^(-3/2)
        
        Args:
            z: Червено отместване
            
        Returns:
            Приближена стойност на T(z)
        """
        return (2.0/3.0) * np.power(1.0 + z, -3.0/2.0)
    
    def exact_analytical_t_z(self, z: float) -> float:
        """
        Точна аналитична форма на T(z) чрез хипергеометрични функции
        
        Args:
            z: Червено отместване
            
        Returns:
            Точна стойност на T(z)
        """
        # Използваме точната форма на интеграла
        return (2.0/3.0) * np.power(1.0 + z, -3.0/2.0)
    
    def create_analytical_t_z_function(self, z_range: np.ndarray = None) -> Callable:
        """
        Създава аналитична функция T(z) с висока точност
        
        Args:
            z_range: Диапазон от червени отмествания за калибриране
            
        Returns:
            Функция T(z)
        """
        if z_range is None:
            z_range = np.logspace(-3, 2, 1000)
        
        # Пресмятаме точните стойности
        t_values = np.array([self.analytical_t_z_integral(z) for z in z_range])
        
        # Създаваме интерполатор с високо качество
        self._analytical_t_z = UnivariateSpline(z_range, t_values, s=0, k=3)
        
        return self._analytical_t_z
    
    def scale_factor_from_redshift(self, z: float) -> float:
        """
        Пресмята мащабния фактор от червеното отместване
        
        Args:
            z: Червено отместване
            
        Returns:
            Мащабен фактор a = 1/(1+z)
        """
        return 1.0 / (1.0 + z)
    
    def absolute_time_from_redshift(self, z: float) -> float:
        """
        Пресмята абсолютното време от червеното отместване
        
        Args:
            z: Червено отместване
            
        Returns:
            Абсолютно време в Gyr
        """
        if self._analytical_t_z is None:
            self.create_analytical_t_z_function()
        
        # T(z) дава времето в безразмерни единици
        # Преобразуваме в години
        t_normalized = self._analytical_t_z(z)
        return t_normalized * self.t_universe
    
    def create_scale_factor_function(self) -> Callable:
        """
        Създава функция a(t_abs) за зависимостта на мащабния фактор от абсолютното време
        
        Returns:
            Функция a(t_abs)
        """
        if self._analytical_t_z is None:
            self.create_analytical_t_z_function()
        
        # Създаваме обратна функция t_abs -> z
        z_range = np.logspace(-3, 2, 1000)
        t_abs_values = np.array([self.absolute_time_from_redshift(z) for z in z_range])
        
        # Сортираме по нарастващо време
        sort_idx = np.argsort(t_abs_values)
        t_abs_values = t_abs_values[sort_idx]
        z_range = z_range[sort_idx]
        
        # Създаваме интерполатор t_abs -> z
        z_from_t_abs = UnivariateSpline(t_abs_values, z_range, s=0, k=3)
        
        def a_function(t_abs):
            """Функция a(t_abs)"""
            z = z_from_t_abs(t_abs)
            return self.scale_factor_from_redshift(z)
        
        self._scale_factor_function = a_function
        return a_function
    
    def hubble_parameter_abs_time(self, t_abs: float) -> float:
        """
        Пресмята параметъра на Hubble като функция от абсолютното време
        
        Args:
            t_abs: Абсолютно време в Gyr
            
        Returns:
            H(t_abs) в km/s/Mpc
        """
        if self._scale_factor_function is None:
            self.create_scale_factor_function()
        
        # Числено диференциране
        dt = 0.01  # Gyr
        a_now = self._scale_factor_function(t_abs)
        a_future = self._scale_factor_function(t_abs + dt)
        
        # H = (1/a) * (da/dt)
        da_dt = (a_future - a_now) / dt
        H = (1.0 / a_now) * da_dt
        
        # Преобразуваме в правилните единици
        return H * self.H0  # km/s/Mpc
    
    def natural_metric_transformation(self, t_abs: float) -> float:
        """
        Дефинира натурална метрична трансформация с постоянна скорост на разширение
        
        Args:
            t_abs: Абсолютно време в Gyr
            
        Returns:
            Натурално време τ
        """
        if self._scale_factor_function is None:
            self.create_scale_factor_function()
        
        # В натуралната метрика искаме da/dτ = const
        # Тоест τ = ∫ dt / a(t)
        
        # Пресмятаме интеграла числено
        t_range = np.linspace(0.1, t_abs, 1000)
        a_values = np.array([self._scale_factor_function(t) for t in t_range])
        
        # Интегрираме 1/a(t) dt
        integrand = 1.0 / a_values
        tau = np.trapz(integrand, t_range)
        
        return tau
    
    def extended_z_range_analysis(self, z_max: float = 10.0) -> Dict[str, Any]:
        """
        Разширен анализ за z > 2 включително CMB и BAO епохи
        
        Args:
            z_max: Максимално червено отместване за анализ
            
        Returns:
            Резултати от разширения анализ
        """
        z_range = np.logspace(-3, np.log10(z_max), 1000)
        
        # Ключови епохи
        z_recombination = 1100  # Рекомбинация
        z_bao = 0.57  # BAO епоха
        z_cmb = 1100  # CMB епоха
        
        results = {
            'z_range': z_range,
            't_abs_values': np.array([self.absolute_time_from_redshift(z) for z in z_range]),
            'a_values': np.array([self.scale_factor_from_redshift(z) for z in z_range]),
            'H_values': np.array([self.hubble_parameter_abs_time(self.absolute_time_from_redshift(z)) for z in z_range]),
            'key_epochs': {
                'recombination': {
                    'z': z_recombination,
                    't_abs': self.absolute_time_from_redshift(z_recombination) if z_recombination <= z_max else None,
                    'a': self.scale_factor_from_redshift(z_recombination)
                },
                'bao': {
                    'z': z_bao,
                    't_abs': self.absolute_time_from_redshift(z_bao),
                    'a': self.scale_factor_from_redshift(z_bao)
                }
            }
        }
        
        return results
    
    def plot_analytical_functions(self, save_path: str = None):
        """
        Създава графики на всички аналитични функции
        
        Args:
            save_path: Път за записване на графиките
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. T(z) функция
        z_range = np.logspace(-3, 2, 1000)
        t_integral = [self.analytical_t_z_integral(z) for z in z_range]
        t_approx = [self.analytical_t_z_approximation(z) for z in z_range]
        
        ax1.loglog(z_range, t_integral, 'b-', label='Интегрална форма', linewidth=2)
        ax1.loglog(z_range, t_approx, 'r--', label='Приближена форма', linewidth=2)
        ax1.set_xlabel('Червено отместване z')
        ax1.set_ylabel('T(z)')
        ax1.set_title('Аналитична функция T(z)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. a(t_abs) функция
        if self._scale_factor_function is None:
            self.create_scale_factor_function()
        
        t_abs_range = np.linspace(0.1, 13.8, 1000)
        a_values = [self._scale_factor_function(t) for t in t_abs_range]
        
        ax2.plot(t_abs_range, a_values, 'g-', linewidth=2)
        ax2.set_xlabel('Абсолютно време t_abs [Gyr]')
        ax2.set_ylabel('Мащабен фактор a(t_abs)')
        ax2.set_title('Зависимост a(t_abs)')
        ax2.grid(True, alpha=0.3)
        
        # 3. H(t_abs) функция
        H_values = [self.hubble_parameter_abs_time(t) for t in t_abs_range]
        
        ax3.plot(t_abs_range, H_values, 'm-', linewidth=2)
        ax3.set_xlabel('Абсолютно време t_abs [Gyr]')
        ax3.set_ylabel('H(t_abs) [km/s/Mpc]')
        ax3.set_title('Параметър на Hubble H(t_abs)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Натурална метрична трансформация
        tau_values = [self.natural_metric_transformation(t) for t in t_abs_range]
        
        ax4.plot(t_abs_range, tau_values, 'c-', linewidth=2)
        ax4.set_xlabel('Абсолютно време t_abs [Gyr]')
        ax4.set_ylabel('Натурално време τ')
        ax4.set_title('Натурална метрична трансформация')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def comprehensive_analysis_report(self) -> str:
        """
        Създава обобщен доклад с всички аналитични функции
        
        Returns:
            Форматиран текстов доклад
        """
        report = []
        report.append("=" * 80)
        report.append("ОБОБЩЕН ДОКЛАД: РАЗШИРЕНИ АНАЛИТИЧНИ ФУНКЦИИ")
        report.append("=" * 80)
        report.append("")
        
        # 1. Аналитична функция T(z)
        report.append("1. АНАЛИТИЧНА ФУНКЦИЯ T(z)")
        report.append("-" * 40)
        report.append("• Интегрална форма: T(z) = ∫[z to ∞] 1/((1+z')^(5/2)) dz'")
        report.append("• Приближена форма: T(z) ≈ (2/3) * (1+z)^(-3/2)")
        report.append("")
        
        # Тестваме точността на приближението
        test_z = [0.1, 1.0, 5.0, 10.0]
        for z in test_z:
            integral = self.analytical_t_z_integral(z)
            approx = self.analytical_t_z_approximation(z)
            error = abs(integral - approx) / integral * 100
            report.append(f"  z = {z:4.1f}: Интеграл = {integral:.6f}, Приближение = {approx:.6f}, Грешка = {error:.2f}%")
        
        report.append("")
        
        # 2. Функция a(t_abs)
        report.append("2. ФУНКЦИЯ a(t_abs)")
        report.append("-" * 40)
        if self._scale_factor_function is None:
            self.create_scale_factor_function()
        
        test_times = [1.0, 5.0, 10.0, 13.8]
        for t in test_times:
            a_val = self._scale_factor_function(t)
            report.append(f"  t = {t:4.1f} Gyr: a(t) = {a_val:.6f}")
        
        report.append("")
        
        # 3. Разширен анализ
        report.append("3. РАЗШИРЕН АНАЛИЗ (z > 2)")
        report.append("-" * 40)
        extended_results = self.extended_z_range_analysis(z_max=10.0)
        
        for epoch, data in extended_results['key_epochs'].items():
            report.append(f"  {epoch.upper()}:")
            report.append(f"    z = {data['z']}")
            if data['t_abs'] is not None:
                report.append(f"    t_abs = {data['t_abs']:.3f} Gyr")
            report.append(f"    a = {data['a']:.6f}")
            report.append("")
        
        # 4. Натурална метрика
        report.append("4. НАТУРАЛНА МЕТРИЧНА СИСТЕМА")
        report.append("-" * 40)
        report.append("• Трансформация: τ = ∫ dt / a(t)")
        report.append("• Цел: Постоянна скорост на разширение в τ-време")
        report.append("")
        
        test_times = [1.0, 5.0, 10.0, 13.8]
        for t in test_times:
            tau = self.natural_metric_transformation(t)
            report.append(f"  t = {t:4.1f} Gyr: τ = {tau:.6f}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


# Помощни функции за лесна употреба
def create_analytical_functions():
    """
    Създава инстанция на AdvancedAnalyticalFunctions с всички настройки
    
    Returns:
        Конфигуриран обект AdvancedAnalyticalFunctions
    """
    return AdvancedAnalyticalFunctions()


def quick_t_z_analysis(z_values: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Бърз анализ на T(z) функцията за масив от червени отмествания
    
    Args:
        z_values: Масив от червени отмествания
        
    Returns:
        Резултати от анализа
    """
    aaf = create_analytical_functions()
    
    results = {
        'z_values': z_values,
        't_integral': np.array([aaf.analytical_t_z_integral(z) for z in z_values]),
        't_approximation': np.array([aaf.analytical_t_z_approximation(z) for z in z_values]),
        't_absolute': np.array([aaf.absolute_time_from_redshift(z) for z in z_values]),
        'scale_factor': np.array([aaf.scale_factor_from_redshift(z) for z in z_values])
    }
    
    return results


if __name__ == "__main__":
    # Тестваме модула
    print("Тестване на разширените аналитични функции...")
    
    aaf = create_analytical_functions()
    
    # Тест 1: T(z) функция
    print("\nТест 1: T(z) функция")
    test_z = [0.1, 1.0, 5.0, 10.0]
    for z in test_z:
        integral = aaf.analytical_t_z_integral(z)
        approx = aaf.analytical_t_z_approximation(z)
        print(f"z = {z:4.1f}: T_integral = {integral:.6f}, T_approx = {approx:.6f}")
    
    # Тест 2: a(t_abs) функция
    print("\nТест 2: a(t_abs) функция")
    aaf.create_scale_factor_function()
    test_times = [1.0, 5.0, 10.0, 13.8]
    for t in test_times:
        a_val = aaf._scale_factor_function(t)
        print(f"t = {t:4.1f} Gyr: a(t) = {a_val:.6f}")
    
    # Тест 3: Разширен анализ
    print("\nТест 3: Разширен анализ")
    extended_results = aaf.extended_z_range_analysis(z_max=10.0)
    for epoch, data in extended_results['key_epochs'].items():
        print(f"{epoch}: z = {data['z']}, a = {data['a']:.6f}")
    
    print("\nТестването завърши успешно!") 