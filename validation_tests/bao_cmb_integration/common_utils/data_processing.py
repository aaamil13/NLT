"""
Модул за обработка на данни за BAO и CMB анализ

Функционалности:
- Обработка на реални BAO данни
- Филтриране и интерполация на CMB данни
- Статистически анализ на грешки
- Корелационни матрици
- Качествен контрол на данните
"""

import numpy as np
from scipy import interpolate, stats
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Валидиране на качеството на данните"""
    
    @staticmethod
    def validate_z_data(z: np.ndarray, z_min: float = 0.0, z_max: float = 10.0) -> bool:
        """
        Валидира данни за червено отместване
        
        Args:
            z: Масив с червени отмествания
            z_min: Минимално допустимо z
            z_max: Максимално допустимо z
            
        Returns:
            True ако данните са валидни
        """
        if not isinstance(z, np.ndarray):
            z = np.array(z)
            
        # Проверки
        checks = [
            len(z) > 0,  # Не е празен
            np.all(np.isfinite(z)),  # Няма NaN/inf
            np.all(z >= z_min),  # В допустимия диапазон
            np.all(z <= z_max),
            np.all(z >= 0)  # Физично разумно
        ]
        
        return all(checks)
    
    @staticmethod
    def validate_measurement_data(data: np.ndarray, errors: np.ndarray) -> bool:
        """
        Валидира измерителни данни с грешки
        
        Args:
            data: Измерени стойности
            errors: Грешки в измерванията
            
        Returns:
            True ако данните са валидни
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if not isinstance(errors, np.ndarray):
            errors = np.array(errors)
            
        # Проверки
        checks = [
            len(data) == len(errors),  # Еднакви размери
            np.all(np.isfinite(data)),  # Няма NaN/inf
            np.all(np.isfinite(errors)),
            np.all(errors > 0),  # Положителни грешки
            np.all(errors < 10 * np.abs(data))  # Разумни грешки
        ]
        
        return all(checks)
    
    @staticmethod
    def detect_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Откриване на статистически отклонения (outliers)
        
        Args:
            data: Данни за анализ
            threshold: Праг в стандартни отклонения
            
        Returns:
            Булев масив с отклоненията
        """
        if len(data) < 3:
            return np.zeros(len(data), dtype=bool)
            
        # Z-score метод
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > threshold
        
        logger.info(f"Открити {np.sum(outliers)} отклонения от {len(data)} точки")
        return outliers


class BAODataProcessor:
    """Процесор за BAO данни"""
    
    def __init__(self):
        self.validator = DataValidator()
        
    def process_bao_measurements(self, z: np.ndarray, D_V_over_rs: np.ndarray, 
                               errors: np.ndarray, filter_outliers: bool = True) -> Dict[str, np.ndarray]:
        """
        Обработва BAO измервания
        
        Args:
            z: Червени отмествания
            D_V_over_rs: Отношение D_V/r_s
            errors: Грешки в измерванията
            filter_outliers: Дали да филтрира отклоненията
            
        Returns:
            Обработени данни
        """
        # Валидиране
        if not self.validator.validate_z_data(z):
            raise ValueError("Невалидни данни за червено отместване")
        if not self.validator.validate_measurement_data(D_V_over_rs, errors):
            raise ValueError("Невалидни измерителни данни")
            
        # Копиране на данните
        z_clean = z.copy()
        D_V_clean = D_V_over_rs.copy()
        err_clean = errors.copy()
        
        # Филтриране на отклонения
        if filter_outliers:
            outliers = self.validator.detect_outliers(D_V_over_rs)
            if np.any(outliers):
                logger.warning(f"Премахнати {np.sum(outliers)} отклонения")
                mask = ~outliers
                z_clean = z_clean[mask]
                D_V_clean = D_V_clean[mask]
                err_clean = err_clean[mask]
        
        # Сортиране по z
        sort_idx = np.argsort(z_clean)
        z_clean = z_clean[sort_idx]
        D_V_clean = D_V_clean[sort_idx]
        err_clean = err_clean[sort_idx]
        
        # Статистики
        stats_info = {
            'N_points': len(z_clean),
            'z_range': [z_clean.min(), z_clean.max()],
            'D_V_range': [D_V_clean.min(), D_V_clean.max()],
            'mean_error': np.mean(err_clean),
            'relative_error': np.mean(err_clean / D_V_clean)
        }
        
        logger.info(f"Обработени BAO данни: {stats_info['N_points']} точки")
        logger.info(f"Z диапазон: {stats_info['z_range'][0]:.3f} - {stats_info['z_range'][1]:.3f}")
        
        return {
            'z': z_clean,
            'D_V_over_rs': D_V_clean,
            'errors': err_clean,
            'statistics': stats_info
        }
    
    def interpolate_bao_data(self, z: np.ndarray, D_V_over_rs: np.ndarray, 
                           errors: np.ndarray, z_target: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Интерполира BAO данни на нова мрежа
        
        Args:
            z: Оригинални червени отмествания
            D_V_over_rs: Оригинални стойности
            errors: Грешки
            z_target: Целева мрежа
            
        Returns:
            Интерполирани данни
        """
        # Интерполация на стойностите
        f_interp = interpolate.interp1d(z, D_V_over_rs, kind='cubic', 
                                      bounds_error=False, fill_value='extrapolate')
        D_V_interp = f_interp(z_target)
        
        # Интерполация на грешките
        f_err = interpolate.interp1d(z, errors, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
        err_interp = f_err(z_target)
        
        # Маскиране на екстраполираните области
        mask = (z_target >= z.min()) & (z_target <= z.max())
        
        logger.info(f"Интерполирани {np.sum(mask)} от {len(z_target)} точки")
        
        return {
            'z': z_target,
            'D_V_over_rs': D_V_interp,
            'errors': err_interp,
            'interpolation_mask': mask
        }
    
    def create_covariance_matrix(self, errors: np.ndarray, 
                               correlation_length: float = 0.1) -> np.ndarray:
        """
        Създава корелационна матрица за BAO данни
        
        Args:
            errors: Грешки в измерванията
            correlation_length: Дължина на корелацията в z
            
        Returns:
            Корелационна матрица
        """
        N = len(errors)
        cov_matrix = np.zeros((N, N))
        
        # Диагонални елементи (дисперсии)
        np.fill_diagonal(cov_matrix, errors**2)
        
        # Извън-диагонални елементи (корелации)
        for i in range(N):
            for j in range(i+1, N):
                correlation = np.exp(-abs(i-j) / correlation_length)
                cov_matrix[i, j] = correlation * errors[i] * errors[j]
                cov_matrix[j, i] = cov_matrix[i, j]
        
        return cov_matrix


class CMBDataProcessor:
    """Процесор за CMB данни"""
    
    def __init__(self):
        self.validator = DataValidator()
        
    def process_cmb_power_spectrum(self, l: np.ndarray, C_l: np.ndarray, 
                                 C_l_err: np.ndarray, l_min: int = 2, 
                                 l_max: int = 2500) -> Dict[str, np.ndarray]:
        """
        Обработва CMB power spectrum
        
        Args:
            l: Мултиполни моменти
            C_l: Power spectrum стойности
            C_l_err: Грешки в power spectrum
            l_min: Минимален l
            l_max: Максимален l
            
        Returns:
            Обработени данни
        """
        # Валидиране
        if not self.validator.validate_measurement_data(C_l, C_l_err):
            raise ValueError("Невалидни CMB данни")
            
        # Филтриране по l диапазон
        mask = (l >= l_min) & (l <= l_max) & (C_l > 0)
        
        l_clean = l[mask]
        C_l_clean = C_l[mask]
        C_l_err_clean = C_l_err[mask]
        
        # Сортиране по l
        sort_idx = np.argsort(l_clean)
        l_clean = l_clean[sort_idx]
        C_l_clean = C_l_clean[sort_idx]
        C_l_err_clean = C_l_err_clean[sort_idx]
        
        # Статистики
        stats_info = {
            'N_points': len(l_clean),
            'l_range': [l_clean.min(), l_clean.max()],
            'C_l_range': [C_l_clean.min(), C_l_clean.max()],
            'mean_relative_error': np.mean(C_l_err_clean / C_l_clean)
        }
        
        logger.info(f"Обработени CMB данни: {stats_info['N_points']} точки")
        logger.info(f"l диапазон: {stats_info['l_range'][0]} - {stats_info['l_range'][1]}")
        
        return {
            'l': l_clean,
            'C_l': C_l_clean,
            'C_l_err': C_l_err_clean,
            'statistics': stats_info
        }
    
    def extract_acoustic_peaks(self, l: np.ndarray, C_l: np.ndarray, 
                             n_peaks: int = 5) -> Dict[str, np.ndarray]:
        """
        Извлича акустични пикове от CMB power spectrum
        
        Args:
            l: Мултиполни моменти
            C_l: Power spectrum стойности
            n_peaks: Брой пикове за извличане
            
        Returns:
            Данни за пиковете
        """
        # Смутиране за намаляване на шума
        from scipy.signal import savgol_filter
        C_l_smooth = savgol_filter(C_l, window_length=11, polyorder=3)
        
        # Намиране на пикове
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(C_l_smooth, height=0.1*C_l_smooth.max(), 
                                     distance=50)
        
        # Сортиране по височина
        peak_heights = C_l_smooth[peaks]
        sort_idx = np.argsort(peak_heights)[::-1]
        
        # Взимане на най-високите пикове
        n_peaks = min(n_peaks, len(peaks))
        best_peaks = peaks[sort_idx[:n_peaks]]
        
        # Сортиране по l
        best_peaks = np.sort(best_peaks)
        
        logger.info(f"Извлечени {n_peaks} акустични пика")
        
        return {
            'l_peaks': l[best_peaks],
            'C_l_peaks': C_l[best_peaks],
            'peak_indices': best_peaks,
            'n_peaks': n_peaks
        }
    
    def binning_cmb_data(self, l: np.ndarray, C_l: np.ndarray, 
                        C_l_err: np.ndarray, bin_size: int = 10) -> Dict[str, np.ndarray]:
        """
        Групира CMB данни в bins за намаляване на шума
        
        Args:
            l: Мултиполни моменти
            C_l: Power spectrum стойности
            C_l_err: Грешки
            bin_size: Размер на bin
            
        Returns:
            Групирани данни
        """
        n_bins = len(l) // bin_size
        
        l_binned = np.zeros(n_bins)
        C_l_binned = np.zeros(n_bins)
        C_l_err_binned = np.zeros(n_bins)
        
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size
            
            # Усреднени стойности
            l_binned[i] = np.mean(l[start:end])
            
            # Претеглено усредняване
            weights = 1 / C_l_err[start:end]**2
            C_l_binned[i] = np.average(C_l[start:end], weights=weights)
            
            # Грешка от претеглено усредняване
            C_l_err_binned[i] = 1 / np.sqrt(np.sum(weights))
        
        logger.info(f"Групирани данни: {len(l)} -> {n_bins} bins")
        
        return {
            'l': l_binned,
            'C_l': C_l_binned,
            'C_l_err': C_l_err_binned,
            'bin_size': bin_size
        }


class StatisticalAnalyzer:
    """Статистически анализатор за данни"""
    
    @staticmethod
    def calculate_chi_squared(theory: np.ndarray, data: np.ndarray, 
                            errors: np.ndarray) -> float:
        """
        Изчислява χ² статистика
        
        Args:
            theory: Теоретични стойности
            data: Наблюдавани стойности
            errors: Грешки в наблюденията
            
        Returns:
            χ² стойност
        """
        residuals = (theory - data) / errors
        chi_squared = np.sum(residuals**2)
        return chi_squared
    
    @staticmethod
    def calculate_reduced_chi_squared(theory: np.ndarray, data: np.ndarray, 
                                   errors: np.ndarray, n_params: int) -> float:
        """
        Изчислява редуциран χ² (χ²/dof)
        
        Args:
            theory: Теоретични стойности
            data: Наблюдавани стойности
            errors: Грешки
            n_params: Брой параметри в модела
            
        Returns:
            Редуциран χ²
        """
        chi_squared = StatisticalAnalyzer.calculate_chi_squared(theory, data, errors)
        dof = len(data) - n_params
        return chi_squared / dof if dof > 0 else float('inf')
    
    @staticmethod
    def calculate_aic(chi_squared: float, n_params: int, n_data: int) -> float:
        """
        Изчислява Akaike Information Criterion
        
        Args:
            chi_squared: χ² стойност
            n_params: Брой параметри
            n_data: Брой точки
            
        Returns:
            AIC стойност
        """
        return chi_squared + 2 * n_params
    
    @staticmethod
    def calculate_bic(chi_squared: float, n_params: int, n_data: int) -> float:
        """
        Изчислява Bayesian Information Criterion
        
        Args:
            chi_squared: χ² стойност
            n_params: Брой параметри
            n_data: Брой точки
            
        Returns:
            BIC стойност
        """
        return chi_squared + n_params * np.log(n_data)
    
    @staticmethod
    def goodness_of_fit_summary(theory: np.ndarray, data: np.ndarray, 
                              errors: np.ndarray, n_params: int) -> Dict[str, float]:
        """
        Обобщение на goodness of fit статистиките
        
        Args:
            theory: Теоретични стойности
            data: Наблюдавани стойности
            errors: Грешки
            n_params: Брой параметри
            
        Returns:
            Речник със статистики
        """
        n_data = len(data)
        chi_squared = StatisticalAnalyzer.calculate_chi_squared(theory, data, errors)
        
        return {
            'chi_squared': chi_squared,
            'reduced_chi_squared': StatisticalAnalyzer.calculate_reduced_chi_squared(
                theory, data, errors, n_params),
            'dof': n_data - n_params,
            'aic': StatisticalAnalyzer.calculate_aic(chi_squared, n_params, n_data),
            'bic': StatisticalAnalyzer.calculate_bic(chi_squared, n_params, n_data),
            'rms_residual': np.sqrt(np.mean(((theory - data) / errors)**2))
        }


def test_data_processing():
    """Тест на модулите за обработка на данни"""
    print("🧪 ТЕСТ НА ОБРАБОТКА НА ДАННИ")
    print("=" * 50)
    
    # Тест данни
    z_test = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
    D_V_test = np.array([7.5, 8.2, 8.9, 9.5, 10.2])
    err_test = np.array([0.2, 0.15, 0.12, 0.18, 0.25])
    
    # BAO процесор
    print("\n📊 BAO ПРОЦЕСОР:")
    bao_processor = BAODataProcessor()
    processed_bao = bao_processor.process_bao_measurements(z_test, D_V_test, err_test)
    print(f"  Обработени точки: {processed_bao['statistics']['N_points']}")
    print(f"  Средна грешка: {processed_bao['statistics']['mean_error']:.3f}")
    
    # CMB процесор
    print("\n🌠 CMB ПРОЦЕСОР:")
    l_test = np.arange(2, 101, 5)
    C_l_test = 5000 * np.exp(-l_test/500) + 100 * np.sin(l_test/50)
    C_l_err_test = 0.1 * C_l_test
    
    cmb_processor = CMBDataProcessor()
    processed_cmb = cmb_processor.process_cmb_power_spectrum(l_test, C_l_test, C_l_err_test)
    print(f"  Обработени l точки: {processed_cmb['statistics']['N_points']}")
    print(f"  Относителна грешка: {processed_cmb['statistics']['mean_relative_error']:.3f}")
    
    # Статистически анализ
    print("\n📈 СТАТИСТИЧЕСКИ АНАЛИЗ:")
    theory_test = D_V_test + 0.1 * np.random.randn(len(D_V_test))
    stats = StatisticalAnalyzer.goodness_of_fit_summary(theory_test, D_V_test, err_test, 2)
    print(f"  χ²: {stats['chi_squared']:.2f}")
    print(f"  Редуциран χ²: {stats['reduced_chi_squared']:.2f}")
    print(f"  AIC: {stats['aic']:.2f}")
    print(f"  BIC: {stats['bic']:.2f}")
    
    print("\n✅ Всички тестове завършиха успешно!")


if __name__ == "__main__":
    test_data_processing() 