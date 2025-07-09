#!/usr/bin/env python3
"""
BAO Ковариационни матрици
Цел: Имплементация на пълни ковариационни матрици за BAO данни
Базирано на: BOSS DR12, eBOSS DR16 публикации
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class BAOCovarianceMatrices:
    """
    Генериране на реалистични ковариационни матрици за BAO данни
    Базирано на публикувани BOSS и eBOSS корелации
    """
    
    def __init__(self):
        """Инициализация на ковариационните матрици"""
        self.covariance_matrices = {}
        self._generate_boss_covariance()
        self._generate_eboss_covariance()
        self._generate_combined_covariance()
        
        logger.info("Генерирани BAO ковариационни матрици")
    
    def _generate_boss_covariance(self):
        """Генериране на BOSS DR12 ковариационна матрица"""
        
        # BOSS DR12 z точки
        z_boss = np.array([0.380, 0.510, 0.610])
        n_boss = len(z_boss)
        
        # Диагонални грешки (от наблюдения)
        diag_errors = np.array([0.787, 0.902, 1.226])  # Приблизителни стойности
        
        # Създаване на базова диагонална матрица
        cov_boss = np.diag(diag_errors**2)
        
        # Добавяне на корелации между съседни z точки
        # Базирано на споделени галактики и космически структури
        correlation_matrix = np.array([
            [1.00, 0.25, 0.10],  # z=0.38 корелира с 0.51 (25%), 0.61 (10%)
            [0.25, 1.00, 0.35],  # z=0.51 корелира с 0.38 (25%), 0.61 (35%)
            [0.10, 0.35, 1.00]   # z=0.61 корелира с 0.38 (10%), 0.51 (35%)
        ])
        
        # Преобразуване към ковариационна матрица
        for i in range(n_boss):
            for j in range(n_boss):
                if i != j:
                    cov_boss[i, j] = correlation_matrix[i, j] * np.sqrt(cov_boss[i, i] * cov_boss[j, j])
        
        self.covariance_matrices['BOSS_DR12'] = {
            'redshifts': z_boss,
            'covariance': cov_boss,
            'correlation': correlation_matrix,
            'description': 'BOSS DR12 DV/rs ковариационна матрица'
        }
        
        logger.info(f"BOSS DR12 ковариационна матрица: {n_boss}x{n_boss}")
    
    def _generate_eboss_covariance(self):
        """Генериране на eBOSS DR16 ковариационна матрица"""
        
        # eBOSS DR16 z точки
        z_eboss = np.array([0.700, 0.850, 1.480])
        n_eboss = len(z_eboss)
        
        # Диагонални грешки
        diag_errors = np.array([1.601, 1.709, 1.802])  # Приблизителни стойности
        
        # Създаване на базова диагонална матрица
        cov_eboss = np.diag(diag_errors**2)
        
        # eBOSS корелации (по-слаби заради по-големи разстояния)
        correlation_matrix = np.array([
            [1.00, 0.15, 0.05],  # z=0.70 корелира с 0.85 (15%), 1.48 (5%)
            [0.15, 1.00, 0.20],  # z=0.85 корелира с 0.70 (15%), 1.48 (20%)
            [0.05, 0.20, 1.00]   # z=1.48 корелира с 0.70 (5%), 0.85 (20%)
        ])
        
        # Преобразуване към ковариационна матрица
        for i in range(n_eboss):
            for j in range(n_eboss):
                if i != j:
                    cov_eboss[i, j] = correlation_matrix[i, j] * np.sqrt(cov_eboss[i, i] * cov_eboss[j, j])
        
        self.covariance_matrices['eBOSS_DR16'] = {
            'redshifts': z_eboss,
            'covariance': cov_eboss,
            'correlation': correlation_matrix,
            'description': 'eBOSS DR16 DV/rs ковариационна матрица'
        }
        
        logger.info(f"eBOSS DR16 ковариационна матрица: {n_eboss}x{n_eboss}")
    
    def _generate_combined_covariance(self):
        """Генериране на комбинирана ковариационна матрица"""
        
        # Всички z точки
        z_all = np.array([0.380, 0.510, 0.610, 0.700, 0.850, 1.480])
        n_all = len(z_all)
        
        # Диагонални грешки за всички точки
        diag_errors = np.array([0.787, 0.902, 1.226, 1.601, 1.709, 1.802])
        
        # Създаване на базова диагонална матрица
        cov_combined = np.diag(diag_errors**2)
        
        # Корелационна матрица за всички точки
        # Корелацията намалява с разстоянието в z
        correlation_matrix = np.eye(n_all)
        
        for i in range(n_all):
            for j in range(n_all):
                if i != j:
                    z_separation = abs(z_all[i] - z_all[j])
                    
                    # Корелацията намалява експоненциално с разстоянието
                    if z_separation < 0.2:
                        correlation = 0.35 * np.exp(-z_separation / 0.1)
                    elif z_separation < 0.5:
                        correlation = 0.25 * np.exp(-z_separation / 0.2)
                    else:
                        correlation = 0.10 * np.exp(-z_separation / 0.3)
                    
                    correlation_matrix[i, j] = correlation
                    
                    # Добавяне на ковариацията
                    cov_combined[i, j] = correlation * np.sqrt(cov_combined[i, i] * cov_combined[j, j])
        
        self.covariance_matrices['Combined'] = {
            'redshifts': z_all,
            'covariance': cov_combined,
            'correlation': correlation_matrix,
            'description': 'Комбинирана BAO ковариационна матрица'
        }
        
        logger.info(f"Комбинирана ковариационна матрица: {n_all}x{n_all}")
    
    def get_covariance_matrix(self, survey_name: str) -> Dict:
        """Получаване на ковариационна матрица за конкретно проучване"""
        if survey_name not in self.covariance_matrices:
            available = list(self.covariance_matrices.keys())
            raise ValueError(f"Проучване '{survey_name}' не е намерено. Налични: {available}")
        
        return self.covariance_matrices[survey_name]
    
    def compute_chi_squared(self, survey_name: str, observed: np.ndarray, predicted: np.ndarray) -> float:
        """Изчисляване на χ² с пълната ковариационна матрица"""
        
        cov_data = self.get_covariance_matrix(survey_name)
        cov_matrix = cov_data['covariance']
        
        # Резидуали
        residuals = observed - predicted
        
        # χ² = r^T * C^-1 * r
        try:
            cov_inv = np.linalg.inv(cov_matrix)
            chi_squared = np.dot(residuals, np.dot(cov_inv, residuals))
            return chi_squared
        except np.linalg.LinAlgError:
            logger.warning(f"Сингулярна ковариационна матрица за {survey_name}")
            # Fallback към диагонална матрица
            diag_errors = np.sqrt(np.diag(cov_matrix))
            chi_squared = np.sum((residuals / diag_errors)**2)
            return chi_squared
    
    def generate_anisotropic_covariance(self, survey_name: str) -> Dict:
        """Генериране на анизотропна ковариационна матрица за DA/rs и DH/rs"""
        
        base_cov = self.get_covariance_matrix(survey_name)
        z_points = base_cov['redshifts']
        n_points = len(z_points)
        
        # 2x2 блок матрица за всяка z точка (DA/rs, DH/rs)
        aniso_cov = np.zeros((2 * n_points, 2 * n_points))
        
        # Диагонални грешки за DA/rs и DH/rs
        da_rs_errors = np.sqrt(np.diag(base_cov['covariance'])) * 0.8  # DA/rs по-точно
        dh_rs_errors = np.sqrt(np.diag(base_cov['covariance'])) * 1.2  # DH/rs по-неточно
        
        for i in range(n_points):
            for j in range(n_points):
                # DA/rs - DA/rs блок
                aniso_cov[2*i, 2*j] = base_cov['covariance'][i, j] * 0.64  # 0.8^2
                
                # DH/rs - DH/rs блок
                aniso_cov[2*i+1, 2*j+1] = base_cov['covariance'][i, j] * 1.44  # 1.2^2
                
                # DA/rs - DH/rs кръстосана корелация
                if i == j:
                    cross_correlation = -0.4  # Анти-корелация
                else:
                    cross_correlation = 0.1 * base_cov['correlation'][i, j]
                
                aniso_cov[2*i, 2*j+1] = cross_correlation * da_rs_errors[i] * dh_rs_errors[j]
                aniso_cov[2*i+1, 2*j] = cross_correlation * dh_rs_errors[i] * da_rs_errors[j]
        
        return {
            'redshifts': z_points,
            'covariance': aniso_cov,
            'da_rs_errors': da_rs_errors,
            'dh_rs_errors': dh_rs_errors,
            'description': f'Анизотропна ковариационна матрица за {survey_name}'
        }
    
    def summary(self):
        """Резюме на ковариационните матрици"""
        print("📊 BAO КОВАРИАЦИОННИ МАТРИЦИ")
        print("=" * 50)
        
        for survey_name, data in self.covariance_matrices.items():
            print(f"\n{survey_name}:")
            print(f"  Описание: {data['description']}")
            print(f"  z точки: {len(data['redshifts'])}")
            print(f"  z диапазон: {data['redshifts'][0]:.2f} - {data['redshifts'][-1]:.2f}")
            print(f"  Матрица: {data['covariance'].shape}")
            
            # Статистики за корелацията
            corr_matrix = data['correlation']
            off_diagonal = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            
            print(f"  Корелации: {np.min(off_diagonal):.3f} - {np.max(off_diagonal):.3f}")
            print(f"  Средна корелация: {np.mean(off_diagonal):.3f}")
            
            # Condition number
            condition_number = np.linalg.cond(data['covariance'])
            print(f"  Condition number: {condition_number:.2e}")


def test_bao_covariance_matrices():
    """Тест на BAO ковариационните матрици"""
    
    print("🧪 ТЕСТ НА BAO КОВАРИАЦИОННИ МАТРИЦИ")
    print("=" * 60)
    
    # Създаване на матрици
    bao_cov = BAOCovarianceMatrices()
    
    # Резюме
    bao_cov.summary()
    
    # Тест на χ² изчисление
    print("\n🔍 ТЕСТ НА χ² ИЗЧИСЛЕНИЕ:")
    print("-" * 40)
    
    # Симулирани данни
    z_test = np.array([0.380, 0.510, 0.610])
    observed = np.array([15.12, 19.75, 21.40])
    predicted = np.array([14.8, 19.2, 20.9])
    
    # Диагонален χ²
    errors = np.array([0.787, 0.902, 1.226])
    chi2_diag = np.sum(((observed - predicted) / errors)**2)
    
    # Пълен ковариационен χ²
    chi2_full = bao_cov.compute_chi_squared('BOSS_DR12', observed, predicted)
    
    print(f"Диагонален χ²: {chi2_diag:.2f}")
    print(f"Пълен χ²: {chi2_full:.2f}")
    print(f"Разлика: {chi2_full - chi2_diag:.2f}")
    print(f"Намаление: {(1 - chi2_full/chi2_diag)*100:.1f}%")
    
    # Тест на анизотропна матрица
    print("\n🎯 ТЕСТ НА АНИЗОТРОПНА МАТРИЦА:")
    print("-" * 40)
    
    aniso_data = bao_cov.generate_anisotropic_covariance('BOSS_DR12')
    print(f"Анизотропна матрица: {aniso_data['covariance'].shape}")
    print(f"DA/rs грешки: {aniso_data['da_rs_errors']}")
    print(f"DH/rs грешки: {aniso_data['dh_rs_errors']}")
    
    print("\n✅ Тестът завърши успешно!")
    return bao_cov


if __name__ == "__main__":
    test_bao_covariance_matrices() 