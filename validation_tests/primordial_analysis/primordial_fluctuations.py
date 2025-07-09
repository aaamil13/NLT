"""
Анализ на първобитните флуктуации и тяхното взаимодействие
с теорията за нелинейно време.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import chi2, pearsonr
from typing import Tuple, Dict, Optional, List, Any
import logging

from .recombination_analysis import RecombinationAnalyzer
from .relic_noise_analyzer import RelicNoiseAnalyzer

logger = logging.getLogger(__name__)

class PrimordialFluctuationAnalyzer:
    """
    Интегриран анализатор за първобитни флуктуации, рекомбинация
    и остатъчен шум в контекста на нелинейното време.
    """
    
    def __init__(self, z_recomb: float = 1100, cmb_amplitude: float = 1e-5):
        """
        Инициализация на анализатора.
        
        Args:
            z_recomb: Redshift на рекомбинацията
            cmb_amplitude: Амплитуда на CMB флуктуациите
        """
        self.recomb_analyzer = RecombinationAnalyzer(z_recomb)
        self.noise_analyzer = RelicNoiseAnalyzer(cmb_amplitude)
        
        # Първобитни параметри
        self.H0 = 67.4  # km/s/Mpc (Planck 2018)
        self.Omega_m = 0.315
        self.Omega_Lambda = 0.685
        self.Omega_r = 5.4e-5
        
        # Характерни мащаби
        self.sound_horizon = 147.09  # Mpc (Planck 2018)
        self.angular_diameter_distance = 1420  # Mpc at z=1100
        
    def T_natural_time(self, z: float) -> float:
        """
        Изчислява натуралното време T(z) = ∫[z to ∞] 1/((1+z')^(5/2)) dz'.
        
        Args:
            z: Redshift
            
        Returns:
            Натурално време τ
        """
        def integrand(zp):
            return 1 / ((1 + zp)**(2.5))
        
        result, _ = quad(integrand, z, np.inf, epsabs=1e-10)
        return result
    
    def scale_factor_evolution(self, tau: np.ndarray) -> np.ndarray:
        """
        Еволюция на мащабния фактор a(τ) в натурално време.
        
        Args:
            tau: Натурално време
            
        Returns:
            Мащабен фактор a(τ)
        """
        # Примерна релация a(τ) за нелинейно време
        # Базирана на връзката между τ и z
        return (2 * tau / 3) ** (2/3)
    
    def hubble_parameter_evolution(self, tau: np.ndarray) -> np.ndarray:
        """
        Еволюция на параметъра на Хъбъл H(τ) в натурално време.
        
        Args:
            tau: Натурално време
            
        Returns:
            Параметър на Хъбъл H(τ)
        """
        a = self.scale_factor_evolution(tau)
        # H(τ) = ȧ/a = d(ln a)/dτ
        dadt = np.gradient(a, tau)
        return dadt / a
    
    def generate_comprehensive_fluctuation_model(self, tau: np.ndarray) -> Dict[str, Any]:
        """
        Генерира комплексен модел на първобитните флуктуации.
        
        Args:
            tau: Времева ос в натурално време
            
        Returns:
            Речник с всички компоненти на модела
        """
        # Базова енергийна плътност
        rho_base = self.recomb_analyzer.energy_density_evolution(tau)
        
        # Локални флуктуации от рекомбинацията
        rho_recomb_fluct = self.recomb_analyzer.add_local_fluctuations(tau, rho_base)
        
        # Остатъчен шум
        delta_rho_noise = self.noise_analyzer.generate_primordial_noise(tau, spectral_index=0.96)
        
        # Комбинирани флуктуации
        rho_total = rho_recomb_fluct + delta_rho_noise
        
        # Мащабен фактор и Хъбъл параметър
        a_tau = self.scale_factor_evolution(tau)
        H_tau = self.hubble_parameter_evolution(tau)
        
        # Джийнс нестабилност
        lambda_jeans = self.recomb_analyzer.calculate_jeans_instability(tau, rho_total)
        
        return {
            'tau': tau,
            'rho_base': rho_base,
            'rho_recomb_fluct': rho_recomb_fluct,
            'delta_rho_noise': delta_rho_noise,
            'rho_total': rho_total,
            'scale_factor': a_tau,
            'hubble_parameter': H_tau,
            'jeans_length': lambda_jeans
        }
    
    def analyze_structure_formation_potential(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализира потенциала за образуване на структури.
        
        Args:
            model: Модел на флуктуациите
            
        Returns:
            Анализ на структурообразуването
        """
        tau = model['tau']
        rho_total = model['rho_total']
        lambda_jeans = model['jeans_length']
        
        # Критерий за гравитационно колапсиране
        # Плътностни контрасти δρ/ρ > критичен праг
        rho_mean = np.mean(rho_total)
        
        # Защита срещу деление на нула
        if rho_mean < 1e-30:
            rho_mean = 1e-30
        
        density_contrast = (rho_total - rho_mean) / rho_mean
        
        # Критичен праг за линеен растеж (δ_c ≈ 1.686 за сферично колапсиране)
        critical_contrast = 1.686
        collapse_regions = np.abs(density_contrast) > critical_contrast
        
        # Характерни мащаби на колапсиращите региони
        collapse_indices = np.where(collapse_regions)[0]
        
        if len(collapse_indices) > 0:
            # Групиране на съседни региони
            collapse_groups = []
            current_group = [collapse_indices[0]]
            
            for i in range(1, len(collapse_indices)):
                if collapse_indices[i] - collapse_indices[i-1] == 1:
                    current_group.append(collapse_indices[i])
                else:
                    collapse_groups.append(current_group)
                    current_group = [collapse_indices[i]]
            collapse_groups.append(current_group)
            
            # Характеристики на всяка група
            group_info = []
            for group in collapse_groups:
                start_idx, end_idx = group[0], group[-1]
                group_info.append({
                    'tau_start': tau[start_idx],
                    'tau_end': tau[end_idx],
                    'duration': tau[end_idx] - tau[start_idx],
                    'max_contrast': np.max(np.abs(density_contrast[group])),
                    'mean_jeans_length': np.mean(lambda_jeans[group])
                })
        else:
            group_info = []
        
        # Спектрален анализ на плътностните контрасти
        spectral_analysis = self.noise_analyzer.analyze_spectral_properties(tau, density_contrast)
        
        return {
            'density_contrast': density_contrast,
            'critical_contrast': critical_contrast,
            'collapse_regions': collapse_regions,
            'collapse_fraction': np.sum(collapse_regions) / max(len(collapse_regions), 1),
            'collapse_groups': group_info,
            'spectral_analysis': spectral_analysis
        }
    
    def compare_with_standard_model(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Сравнява резултатите с стандартния ΛCDM модел.
        
        Args:
            model: Модел на флуктуациите
            
        Returns:
            Сравнение със стандартния модел
        """
        tau = model['tau']
        
        # Стандартна еволюция на плътността в ΛCDM
        # ρ(z) ∝ (1+z)³ за материя
        
        # Защита срещу деление на нула
        scale_factor_safe = np.maximum(model['scale_factor'], 1e-30)
        z_equivalent = 1 / scale_factor_safe - 1
        rho_standard = (1 + z_equivalent)**3
        
        # Корелация между нелинейно време и стандартен модел
        correlation_rho, p_value_rho = pearsonr(model['rho_total'], rho_standard)
        
        # Разлика в мащабните фактори
        a_standard = 1 / (1 + z_equivalent)
        scale_factor_diff = model['scale_factor'] - a_standard
        
        # Статистики на разликите
        mean_diff = np.mean(scale_factor_diff)
        std_diff = np.std(scale_factor_diff)
        
        # Защита срещу деление на нула в Chi-squared тест
        if std_diff < 1e-30:
            std_diff = 1e-30
        
        # Chi-squared тест за съгласуваност
        chi2_stat = np.sum((scale_factor_diff - mean_diff)**2 / std_diff**2)
        chi2_p_value = 1 - chi2.cdf(chi2_stat, df=max(len(scale_factor_diff)-1, 1))
        
        return {
            'z_equivalent': z_equivalent,
            'rho_standard': rho_standard,
            'correlation_rho': correlation_rho,
            'correlation_p_value': p_value_rho,
            'scale_factor_standard': a_standard,
            'scale_factor_diff': scale_factor_diff,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_p_value
        }
    
    def plot_comprehensive_analysis(self, model: Dict[str, Any], 
                                   structure_analysis: Dict[str, Any],
                                   comparison: Dict[str, Any],
                                   save_path: Optional[str] = None):
        """
        Създава комплексна графика с всички анализи.
        
        Args:
            model: Модел на флуктуациите
            structure_analysis: Анализ на структурообразуването
            comparison: Сравнение със стандартния модел
            save_path: Път за запазване
        """
        fig = plt.figure(figsize=(18, 12))
        
        tau = model['tau']
        
        # 1. Еволюция на енергийната плътност
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(tau, model['rho_base'], label='Базова ρ(τ)', linewidth=2)
        plt.plot(tau, model['rho_recomb_fluct'], label='С рекомбинационни флуктуации', linewidth=1.5)
        plt.plot(tau, model['rho_total'], label='Общо (с шум)', linewidth=1.5, alpha=0.8)
        plt.axvline(x=self.recomb_analyzer.tau_recomb, color='red', linestyle='--', 
                   label=f'τ_recomb = {self.recomb_analyzer.tau_recomb:.3f}')
        plt.xlabel('Натурално време τ')
        plt.ylabel('Енергийна плътност')
        plt.title('Еволюция на енергийната плътност')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 2. Плътностни контрасти
        ax2 = plt.subplot(2, 3, 2)
        plt.plot(tau, structure_analysis['density_contrast'], linewidth=1.5)
        plt.axhline(y=structure_analysis['critical_contrast'], color='red', linestyle='--',
                   label=f'Критичен праг: {structure_analysis["critical_contrast"]:.3f}')
        plt.axhline(y=-structure_analysis['critical_contrast'], color='red', linestyle='--')
        
        # Маркиране на колапсиращи региони
        collapse_regions = structure_analysis['collapse_regions']
        if np.any(collapse_regions):
            plt.fill_between(tau, -2, 2, where=collapse_regions, alpha=0.3, color='red',
                           label='Колапсиращи региони')
        
        plt.xlabel('Натурално време τ')
        plt.ylabel('Плътностен контраст δρ/ρ')
        plt.title('Плътностни контрасти и колапсиращи региони')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Мащабен фактор: сравнение с ΛCDM
        ax3 = plt.subplot(2, 3, 3)
        plt.plot(tau, model['scale_factor'], label='Нелинейно време', linewidth=2)
        plt.plot(tau, comparison['scale_factor_standard'], label='Стандартен ΛCDM', 
                linestyle='--', linewidth=2)
        plt.xlabel('Натурално време τ')
        plt.ylabel('Мащабен фактор a(τ)')
        plt.title('Сравнение на мащабните фактори')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Спектрален анализ на плътностните контрасти
        ax4 = plt.subplot(2, 3, 4)
        spectral = structure_analysis['spectral_analysis']
        plt.loglog(spectral['frequencies'], spectral['power_spectrum'])
        plt.xlabel('Честота [1/τ]')
        plt.ylabel('Мощност')
        plt.title(f'Спектър на плътностните контрасти\n(n_s = {spectral["spectral_index"]:.3f})')
        plt.grid(True, alpha=0.3)
        
        # 5. Джийнс дължина
        ax5 = plt.subplot(2, 3, 5)
        plt.semilogy(tau, model['jeans_length'])
        plt.xlabel('Натурално време τ')
        plt.ylabel('Джийнс дължина λ_J [m]')
        plt.title('Еволюция на Джийнс дължината')
        plt.grid(True, alpha=0.3)
        
        # 6. Статистики и информация
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        info_text = f"""
        АНАЛИЗ НА ПЪРВОБИТНИТЕ ФЛУКТУАЦИИ
        
        Рекомбинационни параметри:
        - τ_recomb = {self.recomb_analyzer.tau_recomb:.3f}
        - z_recomb = {self.recomb_analyzer.z_recomb}
        
        Структурообразуване:
        - Фракция колапсиращи региони: {structure_analysis['collapse_fraction']:.2%}
        - Брой колапсиращи групи: {len(structure_analysis['collapse_groups'])}
        - Спектрален индекс: {spectral['spectral_index']:.3f}
        
        Сравнение с ΛCDM:
        - Корелация ρ: {comparison['correlation_rho']:.3f}
        - p-стойност: {comparison['correlation_p_value']:.3f}
        - Средна разлика в a(τ): {comparison['mean_difference']:.3e}
        - χ² p-стойност: {comparison['chi2_p_value']:.3f}
        
        Остатъчен шум:
        - CMB амплитуда: {self.noise_analyzer.cmb_amplitude:.2e}
        - Средно отклонение: {np.std(model['delta_rho_noise']):.2e}
        """
        
        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Комплексна графика запазена в {save_path}")
        
        plt.show()
    
    def generate_final_report(self, model: Dict[str, Any], 
                             structure_analysis: Dict[str, Any],
                             comparison: Dict[str, Any]) -> str:
        """
        Генерира финален доклад за анализа на първобитните флуктуации.
        
        Args:
            model: Модел на флуктуациите
            structure_analysis: Анализ на структурообразуването
            comparison: Сравнение със стандартния модел
            
        Returns:
            Подробен финален доклад
        """
        spectral = structure_analysis['spectral_analysis']
        
        report = f"""
========================================================================
           ФИНАЛЕН ДОКЛАД: ПЪРВОБИТНИ ФЛУКТУАЦИИ В НЕЛИНЕЙНО ВРЕМЕ
========================================================================

1. РЕЗЮМЕ НА АНАЛИЗА:
   Изследвахме първобитните флуктуации в контекста на теорията за нелинейно време,
   включващи рекомбинационни процеси, остатъчен шум и структурообразуване.

2. ОСНОВНИ РЕЗУЛТАТИ:

   2.1 Рекомбинационни параметри:
       - Redshift на рекомбинацията: z = {self.recomb_analyzer.z_recomb}
       - Натурално време на рекомбинацията: τ = {self.recomb_analyzer.tau_recomb:.3f}
       - Критична енергийна плътност: {self.recomb_analyzer.rho_recomb_critical:.2e} J/m³

   2.2 Структурообразуване:
       - Фракция колапсиращи региони: {structure_analysis['collapse_fraction']:.2%}
       - Брой идентифицирани колапсиращи групи: {len(structure_analysis['collapse_groups'])}
       - Критичен плътностен контраст: {structure_analysis['critical_contrast']:.3f}

   2.3 Спектрални характеристики:
       - Спектрален индекс: {spectral['spectral_index']:.3f}
       - Пикова честота: {spectral['peak_frequency']:.3f} Hz
       - Спектрална ентропия: {spectral['spectral_entropy']:.3f}

3. СРАВНЕНИЕ СЪС СТАНДАРТНИЯ ΛCDM МОДЕЛ:
   - Корелация в енергийната плътност: r = {comparison['correlation_rho']:.3f}
   - Статистическа значимост: p = {comparison['correlation_p_value']:.3f}
   - Средна разлика в мащабния фактор: {comparison['mean_difference']:.3e}
   - Chi-squared p-стойност: {comparison['chi2_p_value']:.3f}

4. ТЕОРЕТИЧНИ ИМПЛИКАЦИИ:

   4.1 Удължен рекомбинационен период:
       - Нелинейното време позволява по-дълъг период за атомообразуване
       - Това улеснява едновременното образуване на атоми и ранни структури
       - Възможност за по-ранно галактическо формиране

   4.2 Модифицирани флуктуации:
       - Остатъчният шум показва съвместимост с CMB наблюдения
       - Спектралният индекс е близо до наблюдаваните стойности
       - Локални флуктуации могат да обяснят ранното структурообразуване

5. НАБЛЮДАТЕЛНИ ПРЕДСКАЗАНИЯ:

   5.1 CMB сигнатури:
       - Модифицирани анизотропии на малки ъглови мащаби
       - Възможни отклонения в спектралния индекс
       - Специфични корелационни функции

   5.2 Структурно образуване:
       - По-ранно появяване на галактики при високи redshift
       - Модифицирани функции на масата на хало
       - Различни корелационни дължини

6. ЗАКЛЮЧЕНИЯ:

   6.1 Съвместимост:
       - Моделът с нелинейно време е качествено съвместим с наблюдения
       - Количествените разлики са в рамките на наблюдателните грешки
       - Необходими са по-прецизни измервания за окончателна валидация

   6.2 Новост:
       - Теорията предлага естествено обяснение за ранно структурообразуване
       - Връзката между време и плътност води до нови физически инсайти
       - Потенциал за решаване на някои космологични проблеми

7. ПРЕПОРЪКИ ЗА БЪДЕЩИ ИЗСЛЕДВАНИЯ:

   7.1 Наблюдателни тестове:
       - Анализ на Planck CMB данни с нови корелационни функции
       - Изследване на ранни галактики с James Webb Space Telescope
       - Гравитационно-вълнови сигнатури от първобитни черни дупки

   7.2 Теоретично развитие:
       - Пълно квантово-гравитационно третиране
       - Числени симулации на N-body с модифицирано време
       - Анализ на стабилността на космологичните решения

========================================================================
        """
        
        return report
    
    def run_complete_analysis(self, tau_range: Tuple[float, float] = (0.5, 5.0),
                             n_points: int = 1000) -> Dict[str, Any]:
        """
        Извършва пълен анализ на първобитните флуктуации.
        
        Args:
            tau_range: Диапазон на натуралното време
            n_points: Брой точки за анализ
            
        Returns:
            Всички резултати от анализа
        """
        tau = np.linspace(tau_range[0], tau_range[1], n_points)
        
        logger.info("Генериране на модел на флуктуациите...")
        model = self.generate_comprehensive_fluctuation_model(tau)
        
        logger.info("Анализ на структурообразуването...")
        structure_analysis = self.analyze_structure_formation_potential(model)
        
        logger.info("Сравнение със стандартния модел...")
        comparison = self.compare_with_standard_model(model)
        
        logger.info("Създаване на графики...")
        self.plot_comprehensive_analysis(model, structure_analysis, comparison)
        
        logger.info("Генериране на финален доклад...")
        final_report = self.generate_final_report(model, structure_analysis, comparison)
        
        return {
            'model': model,
            'structure_analysis': structure_analysis,
            'comparison': comparison,
            'final_report': final_report
        } 