"""
Общи утилити за валидационни тестове на теорията за нелинейно време
================================================================

Този модул предоставя общи функции за:
- Оптимизационни методи (Differential Evolution, Basinhopping)
- MCMC и Байесов анализ
- Статистическа значимост
- Обработка на сурови данни

Автор: Система за анализ на нелинейно време
"""

from .optimization_engines import *
from .mcmc_bayesian import *
from .statistical_tests import *
from .data_processors import *

__all__ = [
    'DifferentialEvolutionOptimizer',
    'BasinhoppingOptimizer',
    'MCMCBayesianAnalyzer',
    'StatisticalSignificanceTest',
    'RawDataProcessor'
] 