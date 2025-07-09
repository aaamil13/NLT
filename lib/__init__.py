"""
Библиотека за нелинейно време в космологията
Модули за моделиране на абсолютни координатни системи (АКС)
"""

# Версия на библиотеката
__version__ = "1.0.0"
__author__ = "Нелинейно време в космологията"
__description__ = "Библиотека за моделиране на абсолютни координатни системи"

# Импорти ще се правят при поискване
def get_cosmology_classes():
    """Връща класовете от основния космологичен модул"""
    from .nonlinear_time_cosmology import (
        CosmologicalParameters,
        AbsoluteCoordinateSystem,
        RelativeCoordinateSystem,
        ExpansionCalculator,
        CosmologyVisualizer
    )
    return {
        'CosmologicalParameters': CosmologicalParameters,
        'AbsoluteCoordinateSystem': AbsoluteCoordinateSystem,
        'RelativeCoordinateSystem': RelativeCoordinateSystem,
        'ExpansionCalculator': ExpansionCalculator,
        'CosmologyVisualizer': CosmologyVisualizer
    }

def get_redshift_classes():
    """Връща класовете за redshift калибрация"""
    from .redshift_calibration import (
        LinearTimeStepGenerator,
        RedshiftCalculator,
        ExpansionRateCalibrator,
        RedshiftComparisonVisualizer
    )
    return {
        'LinearTimeStepGenerator': LinearTimeStepGenerator,
        'RedshiftCalculator': RedshiftCalculator,
        'ExpansionRateCalibrator': ExpansionRateCalibrator,
        'RedshiftComparisonVisualizer': RedshiftComparisonVisualizer
    }

def get_real_data_classes():
    """Връща класовете за работа с реални данни"""
    from .real_data_acs_analysis import (
        PantheonDataLoader,
        UnifiedACSFinder,
        LinearACSGenerator,
        LinearExpansionAnalyzer,
        RealDataACSVisualizer
    )
    return {
        'PantheonDataLoader': PantheonDataLoader,
        'UnifiedACSFinder': UnifiedACSFinder,
        'LinearACSGenerator': LinearACSGenerator,
        'LinearExpansionAnalyzer': LinearExpansionAnalyzer,
        'RealDataACSVisualizer': RealDataACSVisualizer
    }

def get_acs_transformation_classes():
    """Връща класовете за АКС трансформация"""
    from .acs_time_transformation import (
        TimeTransformationModel,
        RedshiftTimeRelation,
        ExpansionAnalyzer,
        ExpansionVisualizer
    )
    return {
        'TimeTransformationModel': TimeTransformationModel,
        'RedshiftTimeRelation': RedshiftTimeRelation,
        'ExpansionAnalyzer': ExpansionAnalyzer,
        'ExpansionVisualizer': ExpansionVisualizer
    }

# Удобни функции за директен импорт
def import_cosmology():
    """Импортира основните космологични класове"""
    return get_cosmology_classes()

def import_redshift():
    """Импортира класовете за redshift калибрация"""
    return get_redshift_classes()

def import_real_data():
    """Импортира класовете за работа с реални данни"""
    return get_real_data_classes()

def import_acs_transformation():
    """Импортира класовете за АКС трансформация"""
    return get_acs_transformation_classes()

# Lazy imports - само при поискване
def __getattr__(name):
    """Lazy import на класовете"""
    
    # Основни космологични класове
    if name in ['CosmologicalParameters', 'AbsoluteCoordinateSystem', 
                'RelativeCoordinateSystem', 'ExpansionCalculator', 'CosmologyVisualizer']:
        classes = get_cosmology_classes()
        return classes[name]
    
    # Redshift калибрация
    if name in ['LinearTimeStepGenerator', 'RedshiftCalculator', 
                'ExpansionRateCalibrator', 'RedshiftComparisonVisualizer']:
        classes = get_redshift_classes()
        return classes[name]
    
    # Реални данни
    if name in ['PantheonDataLoader', 'UnifiedACSFinder', 'LinearACSGenerator',
                'LinearExpansionAnalyzer', 'RealDataACSVisualizer']:
        classes = get_real_data_classes()
        return classes[name]
    
    # АКС трансформация
    if name in ['TimeTransformationModel', 'RedshiftTimeRelation',
                'ExpansionAnalyzer', 'ExpansionVisualizer']:
        classes = get_acs_transformation_classes()
        return classes[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 