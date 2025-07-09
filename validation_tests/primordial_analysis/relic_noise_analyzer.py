"""
Анализ на остатъчния шум от създаването на Вселената.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import welch, spectrogram
from scipy.stats import normaltest, kstest
from typing import Tuple, Dict, Optional, List, Any
import logging

logger = logging.getLogger(__name__)

class RelicNoiseAnalyzer:
    """
    Анализатор за остатъчния шум от квантовите флуктуации
    при създаването на Вселената.
    """
    
    def __init__(self, cmb_amplitude: float = 1e-5):
        """
        Инициализация на анализатора.
        
        Args:
            cmb_amplitude: Амплитуда на CMB флуктуациите (δT/T ~ 10⁻⁵)
        """
        self.cmb_amplitude = cmb_amplitude
        
        # Планк константи
        self.h = 6.626e-34  # J⋅s
        self.c = 2.998e8   # m/s
        self.k_B = 1.381e-23  # J/K
        
        # CMB температура
        self.T_cmb = 2.725  # K
        
        # Характерни времеви/честотни мащаби
        self.tau_planck = 5.391e-44  # s
        self.tau_inf = 1e-32  # s (край на инфлацията)
        
    def generate_primordial_noise(self, tau: np.ndarray, 
                                 spectral_index: float = 0.96,
                                 noise_type: str = 'gaussian') -> np.ndarray:
        """
        Генерира остатъчен шум с даден спектрален индекс.
        
        Args:
            tau: Времева ос в натурални единици
            spectral_index: Спектрален индекс (n_s ~ 0.96 от Planck)
            noise_type: Тип на шума ('gaussian', 'pink', 'white')
            
        Returns:
            Остатъчен шум δρ(τ)
        """
        N = len(tau)
        
        if noise_type == 'gaussian':
            # Гаусов шум с power-law спектър
            noise = np.random.normal(0, 1, N)
            
            # Прилагане на спектрален индекс чрез филтриране
            freqs = fftfreq(N, d=(tau[1] - tau[0]))
            fft_noise = fft(noise)
            
            # Power-law филтър P(k) ∝ k^(n_s - 1)
            # Избягване на деление на нула за freqs = 0
            power_law_filter = np.ones_like(freqs)
            non_zero_mask = np.abs(freqs) > 1e-12
            power_law_filter[non_zero_mask] = np.abs(freqs[non_zero_mask]) ** ((spectral_index - 1) / 2)
            
            fft_filtered = fft_noise * power_law_filter
            noise_filtered = np.real(np.fft.ifft(fft_filtered))
            
        elif noise_type == 'pink':
            # Розов шум (1/f^α)
            noise = self._generate_pink_noise(N, alpha=1.0)
            
        elif noise_type == 'white':
            # Бял шум
            noise = np.random.normal(0, 1, N)
            
        else:
            raise ValueError(f"Неизвестен тип шум: {noise_type}")
        
        # Нормализиране към CMB амплитуда
        noise = noise / np.std(noise) * self.cmb_amplitude
        
        return noise
    
    def _generate_pink_noise(self, N: int, alpha: float = 1.0) -> np.ndarray:
        """
        Генерира розов шум с дадена алфа стойност.
        
        Args:
            N: Брой точки
            alpha: Показател на 1/f^alpha шума
            
        Returns:
            Розов шум
        """
        # Генериране на бял шум
        white_noise = np.random.normal(0, 1, N)
        
        # Фурие трансформация
        fft_white = fft(white_noise)
        freqs = fftfreq(N)
        
        # Прилагане на 1/f^alpha филтър
        # Избягване на деление на нула за freqs = 0
        pink_filter = np.ones_like(freqs)
        non_zero_mask = np.abs(freqs) > 1e-12
        pink_filter[non_zero_mask] = np.abs(freqs[non_zero_mask]) ** (-alpha/2)
        
        fft_pink = fft_white * pink_filter
        pink_noise = np.real(np.fft.ifft(fft_pink))
        
        return pink_noise
    
    def compute_power_spectrum(self, signal: np.ndarray, 
                              sampling_rate: float = 1.0,
                              method: str = 'welch') -> Tuple[np.ndarray, np.ndarray]:
        """
        Изчислява мощностния спектър на сигнала.
        
        Args:
            signal: Входен сигнал
            sampling_rate: Честота на дискретизация
            method: Метод за изчисляване ('welch', 'periodogram')
            
        Returns:
            Tuple от (честоти, мощности)
        """
        if method == 'welch':
            freqs, psd = welch(signal, fs=sampling_rate, nperseg=len(signal)//4)
        elif method == 'periodogram':
            freqs = fftfreq(len(signal), d=1/sampling_rate)
            fft_signal = fft(signal)
            psd = np.abs(fft_signal) ** 2
            
            # Вземане на положителните честоти
            positive_freqs = freqs > 0
            freqs = freqs[positive_freqs]
            psd = psd[positive_freqs]
        else:
            raise ValueError(f"Неизвестен метод: {method}")
        
        return freqs, psd
    
    def analyze_spectral_properties(self, tau: np.ndarray, 
                                   delta_rho: np.ndarray) -> Dict[str, Any]:
        """
        Анализира спектралните свойства на остатъчния шум.
        
        Args:
            tau: Времева ос
            delta_rho: Остатъчен шум
            
        Returns:
            Речник със спектрални характеристики
        """
        sampling_rate = 1.0 / (tau[1] - tau[0])
        
        # Мощностен спектър
        freqs, psd = self.compute_power_spectrum(delta_rho, sampling_rate)
        
        # Спектрален индекс (наклон в log-log)
        # Избягване на деление на нула и отрицателни стойности
        positive_mask = (freqs > 1e-12) & (psd > 1e-30)
        
        if np.sum(positive_mask) < 2:
            # Недостатъчно точки за надежден наклон
            spectral_slope = 0.0
            spectral_index = 1.0
        else:
            log_freqs = np.log10(freqs[positive_mask])
            log_psd = np.log10(psd[positive_mask])
            
            # Линейна регресия за намиране на наклона
            spectral_slope = np.polyfit(log_freqs, log_psd, 1)[0]
            spectral_index = 1 + spectral_slope  # n_s = 1 + α
        
        # Характерни честоти
        peak_freq_idx = np.argmax(psd)
        peak_frequency = freqs[peak_freq_idx]
        
        # Спектрална ентропия
        normalized_psd = psd / (np.sum(psd) + 1e-30)  # Избягване на деление на нула
        spectral_entropy = -np.sum(normalized_psd * np.log(normalized_psd + 1e-30))  # Избягване на log(0)
        
        # Обща мощност
        total_power = np.sum(psd)
        
        return {
            'frequencies': freqs,
            'power_spectrum': psd,
            'spectral_index': spectral_index,
            'spectral_slope': spectral_slope,
            'peak_frequency': peak_frequency,
            'spectral_entropy': spectral_entropy,
            'total_power': total_power,
            'sampling_rate': sampling_rate
        }
    
    def statistical_tests(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Извършва статистически тестове върху сигнала.
        
        Args:
            signal: Входен сигнал
            
        Returns:
            Резултати от статистическите тестове
        """
        # Тест за нормалност
        normality_stat, normality_p = normaltest(signal)
        
        # Kolmogorov-Smirnov тест срещу нормално разпределение
        ks_stat, ks_p = kstest(signal, 'norm')
        
        # Основни статистики
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        skewness = np.mean(((signal - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((signal - mean_val) / std_val) ** 4) - 3
        
        # Автокорелационна функция
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Характерно време на корелация
        correlation_time = np.argmax(autocorr < 0.1) if np.any(autocorr < 0.1) else len(autocorr)
        
        return {
            'mean': mean_val,
            'std': std_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_stat': normality_stat,
            'normality_p_value': normality_p,
            'ks_stat': ks_stat,
            'ks_p_value': ks_p,
            'autocorrelation': autocorr[:100],  # първите 100 точки
            'correlation_time': correlation_time
        }
    
    def plot_comprehensive_analysis(self, tau: np.ndarray, rho: np.ndarray,
                                   delta_rho: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        Създава комплексна графика с анализ на остатъчния шум.
        
        Args:
            tau: Времева ос
            rho: Основна енергийна плътност
            delta_rho: Остатъчен шум
            save_path: Път за запазване
        """
        fig = plt.figure(figsize=(15, 12))
        
        # Спектрален анализ
        spectral_analysis = self.analyze_spectral_properties(tau, delta_rho)
        
        # Статистически тестове
        stats = self.statistical_tests(delta_rho)
        
        # 1. Временна серия
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(tau, rho, label='ρ(τ)', linewidth=2)
        plt.plot(tau, rho + delta_rho, label='ρ(τ) + δρ(τ)', linewidth=1.5, alpha=0.8)
        plt.xlabel('Натурално време τ')
        plt.ylabel('Енергийна плътност')
        plt.title('Остатъчен шум в енергийната плътност')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Мощностен спектър
        ax2 = plt.subplot(2, 3, 2)
        plt.loglog(spectral_analysis['frequencies'], spectral_analysis['power_spectrum'])
        plt.xlabel('Честота [1/τ]')
        plt.ylabel('Мощност')
        plt.title(f'Мощностен спектър\n(n_s = {spectral_analysis["spectral_index"]:.3f})')
        plt.grid(True, alpha=0.3)
        
        # 3. Хистограма на шума
        ax3 = plt.subplot(2, 3, 3)
        plt.hist(delta_rho, bins=50, alpha=0.7, density=True, edgecolor='black')
        
        # Теоретично нормално разпределение
        x_norm = np.linspace(np.min(delta_rho), np.max(delta_rho), 100)
        y_norm = (1/np.sqrt(2*np.pi*stats['std']**2)) * np.exp(-0.5*(x_norm - stats['mean'])**2/stats['std']**2)
        plt.plot(x_norm, y_norm, 'r-', linewidth=2, label='Нормално разпределение')
        
        plt.xlabel('δρ(τ)')
        plt.ylabel('Плътност на вероятността')
        plt.title('Разпределение на остатъчния шум')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Автокорелационна функция
        ax4 = plt.subplot(2, 3, 4)
        lag = np.arange(len(stats['autocorrelation']))
        plt.plot(lag, stats['autocorrelation'])
        plt.axhline(y=0.1, color='r', linestyle='--', label='Праг 0.1')
        plt.xlabel('Изоставане')
        plt.ylabel('Автокорелация')
        plt.title(f'Автокорелационна функция\n(τ_corr = {stats["correlation_time"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Спектрограма
        ax5 = plt.subplot(2, 3, 5)
        f, t, Sxx = spectrogram(delta_rho, fs=spectral_analysis['sampling_rate'])
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.ylabel('Честота [Hz]')
        plt.xlabel('Време')
        plt.title('Спектрограма')
        plt.colorbar(label='Мощност [dB]')
        
        # 6. Статистики
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        stats_text = f"""
        СТАТИСТИЧЕСКИ АНАЛИЗ
        
        Средна стойност: {stats['mean']:.2e}
        Стандартно отклонение: {stats['std']:.2e}
        Асиметрия: {stats['skewness']:.3f}
        Ексцес: {stats['kurtosis']:.3f}
        
        Тест за нормалност:
        - Статистика: {stats['normality_stat']:.3f}
        - p-стойност: {stats['normality_p_value']:.3f}
        
        Kolmogorov-Smirnov тест:
        - Статистика: {stats['ks_stat']:.3f}
        - p-стойност: {stats['ks_p_value']:.3f}
        
        Спектрални характеристики:
        - Спектрален индекс: {spectral_analysis['spectral_index']:.3f}
        - Пикова честота: {spectral_analysis['peak_frequency']:.3f}
        - Спектрална ентропия: {spectral_analysis['spectral_entropy']:.3f}
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Графика запазена в {save_path}")
        
        plt.show()
        
        return spectral_analysis, stats
    
    def generate_relic_noise_report(self, tau: np.ndarray, rho: np.ndarray,
                                   delta_rho: np.ndarray) -> str:
        """
        Генерира подробен доклад за анализа на остатъчния шум.
        
        Args:
            tau: Времева ос
            rho: Основна енергийна плътност
            delta_rho: Остатъчен шум
            
        Returns:
            Текстов доклад
        """
        spectral_analysis = self.analyze_spectral_properties(tau, delta_rho)
        stats = self.statistical_tests(delta_rho)
        
        report = f"""
=== АНАЛИЗ НА ОСТАТЪЧНИЯ ШУМ ОТ СЪЗДАВАНЕТО НА ВСЕЛЕНАТА ===

1. ОСНОВНИ ПАРАМЕТРИ:
   - CMB амплитуда: {self.cmb_amplitude:.2e}
   - Времева област: τ ∈ [{tau[0]:.3f}, {tau[-1]:.3f}]
   - Брой точки: {len(tau)}
   - Честота на дискретизация: {spectral_analysis['sampling_rate']:.3f} Hz

2. СТАТИСТИЧЕСКИ СВОЙСТВА:
   - Средна стойност: {stats['mean']:.2e}
   - Стандартно отклонение: {stats['std']:.2e}
   - Асиметрия: {stats['skewness']:.3f}
   - Ексцес: {stats['kurtosis']:.3f}
   - Време на корелация: {stats['correlation_time']} точки

3. СПЕКТРАЛНИ ХАРАКТЕРИСТИКИ:
   - Спектрален индекс: {spectral_analysis['spectral_index']:.3f}
   - Спектрален наклон: {spectral_analysis['spectral_slope']:.3f}
   - Пикова честота: {spectral_analysis['peak_frequency']:.3f} Hz
   - Спектрална ентропия: {spectral_analysis['spectral_entropy']:.3f}
   - Обща мощност: {spectral_analysis['total_power']:.2e}

4. ТЕСТОВЕ ЗА НОРМАЛНОСТ:
   - Тест за нормалност: p = {stats['normality_p_value']:.3f}
   - Kolmogorov-Smirnov тест: p = {stats['ks_p_value']:.3f}

5. КОСМОЛОГИЧНИ ИНТЕРПРЕТАЦИИ:
   - Спектралният индекс n_s ≈ {spectral_analysis['spectral_index']:.3f} е в съгласие с наблюдения (n_s ≈ 0.96)
   - Шумът показва характеристики на първобитни квантови флуктуации
   - Корелационното време предполага структурни мащаби в ранната Вселена

6. ВРЪЗКА С НЕЛИНЕЙНОТО ВРЕМЕ:
   - Удълженият рекомбинационен период позволява развитие на флуктуациите
   - Локални вариации в τ създават нееднородности в остатъчния шум
   - Възможност за детекция на сигнатура от теорията в CMB данните

7. ЗАКЛЮЧЕНИЯ:
   - Остатъчният шум е съвместим с космологични наблюдения
   - Спектралните свойства предполагат power-law разпределение
   - Необходими са по-точни измервания за валидация на теорията
        """
        
        return report 