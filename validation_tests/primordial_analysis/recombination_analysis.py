"""
Анализ на рекомбинацията и образуването на ранни структури
в контекста на нелинейното време.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class RecombinationAnalyzer:
    """
    Анализатор за процеса на рекомбинация и образуването на ранни структури
    в рамките на теорията за нелинейно време.
    """
    
    def __init__(self, z_recomb: float = 1100):
        """
        Инициализация на анализатора.
        
        Args:
            z_recomb: Redshift на рекомбинацията (стандартно z=1100)
        """
        self.z_recomb = z_recomb
        self.a_recomb = 1 / (1 + z_recomb)
        self.tau_recomb = self._calculate_tau_recomb()
        
        # Физически константи
        self.c = 2.998e8  # m/s
        self.h_bar = 1.055e-34  # J⋅s
        self.k_B = 1.381e-23  # J/K
        self.m_e = 9.109e-31  # kg
        self.m_p = 1.673e-27  # kg
        self.alpha = 7.297e-3  # fine structure constant
        
        # Критична енергийна плътност за рекомбинация
        self.rho_recomb_critical = 0.3 * 1.602e-19  # eV/cm³ в J/m³
        
    def _calculate_tau_recomb(self) -> float:
        """Изчислява натуралното време τ при рекомбинацията."""
        def integrand(z):
            return 1 / ((1 + z)**(2.5))
        
        tau, _ = quad(integrand, self.z_recomb, np.inf, epsabs=1e-10)
        return tau
    
    def energy_density_evolution(self, tau: np.ndarray) -> np.ndarray:
        """
        Еволюция на енергийната плътност ρ(τ) в натурално време.
        
        Args:
            tau: Масив от натурални времена
            
        Returns:
            Енергийна плътност като функция от τ
        """
        # Примерна формулировка: ρ(τ) ~ exp(-k*τ)
        # Нормализирано така че при τ_recomb да имаме критичната плътност
        k = 1.5
        A = self.rho_recomb_critical * np.exp(k * self.tau_recomb)
        return A * np.exp(-k * tau)
    
    def add_local_fluctuations(self, tau: np.ndarray, rho_base: np.ndarray,
                              fluct_amplitude: float = 0.1,
                              fluct_centers: Optional[List[float]] = None) -> np.ndarray:
        """
        Добавя локални флуктуации в енергийната плътност.
        
        Args:
            tau: Времева ос
            rho_base: Базова плътност
            fluct_amplitude: Амплитуда на флуктуациите
            fluct_centers: Центрове на флуктуациите
            
        Returns:
            Плътност с добавени флуктуации
        """
        if fluct_centers is None:
            fluct_centers = [self.tau_recomb - 0.2, self.tau_recomb, 
                           self.tau_recomb + 0.2, self.tau_recomb + 0.5]
        
        rho_fluct = rho_base.copy()
        
        for center in fluct_centers:
            width = 0.05
            amplitude = fluct_amplitude * np.max(rho_base)
            fluctuation = amplitude * np.exp(-0.5 * ((tau - center) / width) ** 2)
            rho_fluct += fluctuation
            
        return rho_fluct
    
    def analyze_recombination_window(self, tau: np.ndarray, rho: np.ndarray) -> Dict:
        """
        Анализира прозореца на рекомбинация и идентифицира региони
        с потенциал за ранно образуване на структури.
        
        Args:
            tau: Времева ос
            rho: Енергийна плътност
            
        Returns:
            Речник с анализ на рекомбинационния прозорец
        """
        # Намиране на индексите около рекомбинацията
        recomb_idx = np.argmin(np.abs(tau - self.tau_recomb))
        window_size = 100
        
        start_idx = max(0, recomb_idx - window_size)
        end_idx = min(len(tau), recomb_idx + window_size)
        
        tau_window = tau[start_idx:end_idx]
        rho_window = rho[start_idx:end_idx]
        
        # Продължительност на рекомбинационния прозорец
        recomb_duration = tau_window[-1] - tau_window[0]
        
        # Регион под критичната плътност (атомообразуване)
        atomic_formation_mask = rho_window < self.rho_recomb_critical
        atomic_formation_duration = np.sum(atomic_formation_mask) * (tau_window[1] - tau_window[0])
        
        # Локални минимуми (потенциални галактически центрове)
        local_minima = []
        for i in range(1, len(rho_window) - 1):
            if rho_window[i] < rho_window[i-1] and rho_window[i] < rho_window[i+1]:
                local_minima.append({
                    'tau': tau_window[i],
                    'rho': rho_window[i],
                    'depth': (rho_window[i-1] + rho_window[i+1]) / 2 - rho_window[i]
                })
        
        return {
            'recomb_duration': recomb_duration,
            'atomic_formation_duration': atomic_formation_duration,
            'local_minima': local_minima,
            'fraction_below_critical': np.sum(atomic_formation_mask) / len(atomic_formation_mask),
            'tau_window': tau_window,
            'rho_window': rho_window
        }
    
    def calculate_jeans_instability(self, tau: np.ndarray, rho: np.ndarray, 
                                   temperature: float = 3000) -> np.ndarray:
        """
        Изчислява нестабилността на Jeans за гравитационно колапсиране.
        
        Args:
            tau: Времева ос
            rho: Енергийна плътност
            temperature: Температура в K
            
        Returns:
            Характерна дължина на Jeans λ_J
        """
        # Звукова скорост в газа
        cs = np.sqrt(self.k_B * temperature / self.m_p)
        
        # Дължина на Jeans
        G = 6.674e-11  # m³ kg⁻¹ s⁻²
        
        # Защита срещу деление на нула и отрицателни стойности
        rho_safe = np.maximum(rho, 1e-30)  # Минимална плътност
        
        lambda_jeans = cs * np.sqrt(np.pi / (G * rho_safe))
        
        # Проверка за валидни стойности
        lambda_jeans = np.where(np.isfinite(lambda_jeans), lambda_jeans, 1e30)
        
        return lambda_jeans
    
    def plot_recombination_analysis(self, tau: np.ndarray, rho: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        Визуализира анализа на рекомбинационния процес.
        
        Args:
            tau: Времева ос
            rho: Енергийна плътност
            save_path: Път за запазване на графиката
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Горна графика: Енергийна плътност с флуктуации
        rho_fluct = self.add_local_fluctuations(tau, rho)
        
        ax1.plot(tau, rho, label='ρ(τ) - Средна плътност', linewidth=2, color='blue')
        ax1.plot(tau, rho_fluct, label='ρ(τ) + флуктуации', linewidth=1.5, 
                linestyle='--', color='red')
        ax1.axhline(y=self.rho_recomb_critical, color='green', linestyle=':', 
                   label=f'Критична плътност за рекомбинация')
        ax1.axvline(x=self.tau_recomb, color='orange', linestyle=':', 
                   label=f'τ_recomb ≈ {self.tau_recomb:.3f}')
        
        ax1.set_xlabel('Натурално време τ')
        ax1.set_ylabel('Енергийна плътност ρ(τ) [J/m³]')
        ax1.set_title('Еволюция на енергийната плътност по време на рекомбинация')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Долна графика: Анализ на рекомбинационния прозорец
        analysis = self.analyze_recombination_window(tau, rho_fluct)
        
        ax2.plot(analysis['tau_window'], analysis['rho_window'], 
                linewidth=2, color='purple')
        ax2.axhline(y=self.rho_recomb_critical, color='green', linestyle=':', 
                   label='Критична плътност')
        
        # Маркиране на локални минимуми
        for minimum in analysis['local_minima']:
            ax2.plot(minimum['tau'], minimum['rho'], 'ro', markersize=8)
            ax2.annotate(f'Галактически център?', 
                        xy=(minimum['tau'], minimum['rho']),
                        xytext=(minimum['tau'] + 0.1, minimum['rho'] * 2),
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        ax2.set_xlabel('Натурално време τ')
        ax2.set_ylabel('Енергийна плътност ρ(τ) [J/m³]')
        ax2.set_title('Детайлен анализ на рекомбинационния прозорец')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Добавяне на текстова информация
        info_text = f"""
        Продължителност на рекомбинацията: {analysis['recomb_duration']:.3f} τ
        Време за атомообразуване: {analysis['atomic_formation_duration']:.3f} τ
        Фракция под критичната плътност: {analysis['fraction_below_critical']:.2%}
        Брой потенциални галактически центрове: {len(analysis['local_minima'])}
        """
        
        fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Графика запазена в {save_path}")
        
        plt.show()
        
        return analysis
    
    def generate_comprehensive_report(self, tau: np.ndarray, rho: np.ndarray) -> str:
        """
        Генерира подробен доклад за анализа на рекомбинацията.
        
        Args:
            tau: Времева ос
            rho: Енергийна плътност
            
        Returns:
            Текстов доклад
        """
        rho_fluct = self.add_local_fluctuations(tau, rho)
        analysis = self.analyze_recombination_window(tau, rho_fluct)
        
        report = f"""
=== АНАЛИЗ НА РЕКОМБИНАЦИЯТА И РАННОТО ОБРАЗУВАНЕ НА СТРУКТУРИ ===

1. ОСНОВНИ ПАРАМЕТРИ:
   - Redshift на рекомбинацията: z = {self.z_recomb}
   - Натурално време на рекомбинацията: τ = {self.tau_recomb:.3f}
   - Критична енергийна плътност: ρ_crit = {self.rho_recomb_critical:.2e} J/m³

2. РЕКОМБИНАЦИОНЕН ПРОЗОРЕЦ:
   - Общо време на рекомбинация: Δτ = {analysis['recomb_duration']:.3f}
   - Време за атомообразуване: Δτ_atomic = {analysis['atomic_formation_duration']:.3f}
   - Фракция с атомообразуване: {analysis['fraction_below_critical']:.2%}

3. ПОТЕНЦИАЛНИ ГАЛАКТИЧЕСКИ ЦЕНТРОВЕ:
   - Брой идентифицирани центрове: {len(analysis['local_minima'])}
   - Средна дълбочина на минимумите: {np.mean([m['depth'] for m in analysis['local_minima']]) if analysis['local_minima'] else 0:.2e} J/m³

4. ТЕОРЕТИЧНИ ИМПЛИКАЦИИ:
   - Удължената рекомбинация в натурално време позволява образуването на стабилни атоми
   - Локалните флуктуации създават условия за гравитационно колапсиране
   - Възможност за едновременно образуване на атоми и ранни галактически структури

5. ЗАКЛЮЧЕНИЯ:
   - Моделът с нелинейно време предсказва по-дълъг рекомбинационен период
   - Това обяснява възможността за ранно образуване на структури
   - Необходими са наблюдателни тестове за валидация
        """
        
        return report 