"""
GPS релативистични ефекти
========================

Анализ на релативистични ефекти в GPS системи.

Автор: Система за анализ на нелинейно време
"""

import numpy as np
from typing import Dict, Any


class GPSRelativisticEffects:
    """
    Анализатор за релативистични ефекти в GPS
    """
    
    def __init__(self):
        self.effects_data = {}
    
    def analyze_relativistic_effects(self, velocity: np.ndarray, 
                                   potential: np.ndarray) -> Dict[str, Any]:
        """
        Анализира релативистични ефекти
        
        Args:
            velocity: Скорост на сателитите
            potential: Гравитационен потенциал
            
        Returns:
            Анализ на релативистичните ефекти
        """
        c = 299792458  # м/с
        
        # Специална релативност
        sr_effect = -velocity**2 / (2 * c**2)
        
        # Обща релативност
        gr_effect = potential / c**2
        
        return {
            'special_relativity': sr_effect,
            'general_relativity': gr_effect,
            'total_effect': sr_effect + gr_effect,
            'nonlinear_correction': self._calculate_nonlinear_correction(sr_effect + gr_effect)
        }
    
    def _calculate_nonlinear_correction(self, total_effect: np.ndarray) -> np.ndarray:
        """
        Пресмята нелинейна корекция
        
        Args:
            total_effect: Общ релативистичен ефект
            
        Returns:
            Нелинейна корекция
        """
        # Опростена нелинейна корекция
        return total_effect * 1e-6 * np.abs(total_effect) 