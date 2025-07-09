"""
Тестове за библиотеката за нелинейно време космология
"""

import unittest
import numpy as np
from nonlinear_time_cosmology import *

class TestCosmologicalParameters(unittest.TestCase):
    """Тестове за космологичните параметри"""
    
    def test_default_parameters(self):
        """Тест за правилни стойности по подразбиране"""
        params = CosmologicalParameters()
        
        self.assertEqual(params.initial_density, 1e30)
        self.assertEqual(params.current_density, 2.775e-27)
        self.assertEqual(params.linear_expansion_rate, 1.0)
        self.assertEqual(params.time_scaling_exponent, 0.5)
        self.assertEqual(params.universe_age_abs, 13.8e9)
        self.assertEqual(params.universe_age_rel, 13.8e9)
    
    def test_custom_parameters(self):
        """Тест за персонализирани параметри"""
        params = CosmologicalParameters(
            initial_density=5e29,
            linear_expansion_rate=2.0,
            time_scaling_exponent=0.3
        )
        
        self.assertEqual(params.initial_density, 5e29)
        self.assertEqual(params.linear_expansion_rate, 2.0)
        self.assertEqual(params.time_scaling_exponent, 0.3)

class TestAbsoluteCoordinateSystem(unittest.TestCase):
    """Тестове за абсолютната координатна система"""
    
    def setUp(self):
        """Подготовка за тестовете"""
        self.params = CosmologicalParameters()
    
    def test_acs_creation(self):
        """Тест за създаване на АКС"""
        time_abs = 5e9
        acs = AbsoluteCoordinateSystem(time_abs, self.params)
        
        self.assertEqual(acs.time_abs, time_abs)
        self.assertEqual(acs.params, self.params)
        self.assertIsInstance(acs.scale_factor, float)
        self.assertIsInstance(acs.density, float)
        self.assertIsInstance(acs.time_rate, float)
    
    def test_linear_scale_factor(self):
        """Тест за линеен мащабен фактор в АКС"""
        times = [1e9, 2e9, 3e9, 4e9, 5e9]
        scale_factors = []
        
        for t in times:
            acs = AbsoluteCoordinateSystem(t, self.params)
            scale_factors.append(acs.scale_factor)
        
        # Проверка за линейност - отношението трябва да е постоянно
        ratios = [scale_factors[i+1] / scale_factors[i] for i in range(len(scale_factors)-1)]
        
        # Всички отношения трябва да са приблизително равни
        for i in range(len(ratios)-1):
            self.assertAlmostEqual(ratios[i], ratios[i+1], places=10)
    
    def test_density_calculation(self):
        """Тест за изчисляване на плътност"""
        acs1 = AbsoluteCoordinateSystem(1e9, self.params)
        acs2 = AbsoluteCoordinateSystem(2e9, self.params)
        
        # Плътността трябва да намалява с времето
        self.assertGreater(acs1.density, acs2.density)
        
        # Плътността трябва да намалява обратно пропорционално на a³
        expected_ratio = (acs2.scale_factor / acs1.scale_factor)**3
        actual_ratio = acs1.density / acs2.density
        
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=10)
    
    def test_time_rate_calculation(self):
        """Тест за изчисляване на темп на време"""
        acs = AbsoluteCoordinateSystem(5e9, self.params)
        
        # Темпът на време трябва да е положителен
        self.assertGreater(acs.time_rate, 0)
        
        # Темпът на време трябва да зависи от плътността
        expected_rate = (acs.density / self.params.current_density)**(-self.params.time_scaling_exponent)
        self.assertAlmostEqual(acs.time_rate, expected_rate, places=10)
    
    def test_get_coordinates(self):
        """Тест за получаване на координати"""
        acs = AbsoluteCoordinateSystem(5e9, self.params)
        
        # Координатите трябва да са numpy array с 3 елемента
        coords = acs.get_coordinates("test_object")
        self.assertIsInstance(coords, np.ndarray)
        self.assertEqual(len(coords), 3)
        
        # Същият обект трябва да дава същите координати
        coords2 = acs.get_coordinates("test_object")
        np.testing.assert_array_equal(coords, coords2)
        
        # Различни обекти трябва да дават различни координати
        coords3 = acs.get_coordinates("other_object")
        self.assertFalse(np.array_equal(coords, coords3))

class TestRelativeCoordinateSystem(unittest.TestCase):
    """Тестове за релативната координатна система"""
    
    def setUp(self):
        """Подготовка за тестовете"""
        self.params = CosmologicalParameters()
    
    def test_rcs_creation(self):
        """Тест за създаване на РКС"""
        obs_time = 10e9
        rcs = RelativeCoordinateSystem(obs_time, self.params)
        
        self.assertEqual(rcs.observation_time, obs_time)
        self.assertEqual(rcs.params, self.params)
        self.assertIsInstance(rcs.scale_factor, float)
    
    def test_nonlinear_scale_factor(self):
        """Тест за нелинеен мащабен фактор в РКС"""
        times = [1e9, 2e9, 3e9, 4e9, 5e9]
        scale_factors = []
        
        for t in times:
            rcs = RelativeCoordinateSystem(t, self.params)
            scale_factors.append(rcs.scale_factor)
        
        # Проверка за нелинейност - отношението трябва да се променя
        ratios = [scale_factors[i+1] / scale_factors[i] for i in range(len(scale_factors)-1)]
        
        # Отношенията трябва да нарастват (кубично разширение)
        for i in range(len(ratios)-1):
            self.assertGreater(ratios[i+1], ratios[i])
    
    def test_coordinate_transformation(self):
        """Тест за трансформация на координати"""
        rcs = RelativeCoordinateSystem(10e9, self.params)
        
        # Тестови координати
        abs_coords = np.array([1.0, 2.0, 3.0])
        abs_time = 5e9
        
        # Трансформация
        rel_coords = rcs.transform_from_abs(abs_coords, abs_time)
        
        # Релативните координати трябва да са по-големи (разширение)
        self.assertGreater(np.linalg.norm(rel_coords), np.linalg.norm(abs_coords))
        
        # Трансформацията трябва да запазва формата
        self.assertEqual(len(rel_coords), 3)
        self.assertIsInstance(rel_coords, np.ndarray)
    
    def test_expansion_factor(self):
        """Тест за фактор на разширение"""
        rcs = RelativeCoordinateSystem(10e9, self.params)
        
        # Фактор на разширение трябва да е положителен
        factor = rcs._calculate_expansion_factor(1e9)
        self.assertGreater(factor, 0)
        
        # По-големи времеви измествания трябва да дават по-големи фактори
        factor1 = rcs._calculate_expansion_factor(1e9)
        factor2 = rcs._calculate_expansion_factor(2e9)
        self.assertGreater(factor2, factor1)
    
    def test_redshift_calculation(self):
        """Тест за изчисляване на редшифт"""
        rcs = RelativeCoordinateSystem(10e9, self.params)
        
        # Редшифт трябва да е положителен за позитивни времеви измествания
        z = rcs._calculate_redshift(1e9)
        self.assertGreater(z, 0)
        
        # По-големи времеви измествания трябва да дават по-големи редшифт
        z1 = rcs._calculate_redshift(1e9)
        z2 = rcs._calculate_redshift(2e9)
        self.assertGreater(z2, z1)

class TestExpansionCalculator(unittest.TestCase):
    """Тестове за калкулатора на разширение"""
    
    def setUp(self):
        """Подготовка за тестовете"""
        self.params = CosmologicalParameters()
        self.calculator = ExpansionCalculator(self.params)
    
    def test_abs_expansion_coefficient(self):
        """Тест за коефициент на разширение в АКС"""
        coeff = self.calculator.calculate_abs_expansion_coefficient(1e9, 2e9)
        
        # Коефициентът трябва да е положителен
        self.assertGreater(coeff, 0)
        
        # За линейно разширение, коефициентът трябва да е постоянен
        coeff1 = self.calculator.calculate_abs_expansion_coefficient(1e9, 2e9)
        coeff2 = self.calculator.calculate_abs_expansion_coefficient(2e9, 3e9)
        self.assertAlmostEqual(coeff1, coeff2, places=10)
    
    def test_rel_expansion_coefficient(self):
        """Тест за коефициент на разширение в РКС"""
        coeff = self.calculator.calculate_rel_expansion_coefficient(1e9, 2e9)
        
        # Коефициентът трябва да е положителен
        self.assertGreater(coeff, 0)
        
        # За кубично разширение, коефициентът трябва да нараства
        coeff1 = self.calculator.calculate_rel_expansion_coefficient(1e9, 2e9)
        coeff2 = self.calculator.calculate_rel_expansion_coefficient(2e9, 3e9)
        self.assertGreater(coeff2, coeff1)
    
    def test_linearity_check(self):
        """Тест за проверка на линейност"""
        time_points = [1e9, 2e9, 3e9, 4e9, 5e9]
        
        # Проверка за АКС (трябва да е линейна)
        abs_result = self.calculator.check_linearity(time_points, "abs")
        self.assertTrue(abs_result['is_linear'])
        
        # Проверка за РКС (трябва да е нелинейна)
        rel_result = self.calculator.check_linearity(time_points, "rel")
        self.assertFalse(rel_result['is_linear'])
    
    def test_expansion_comparison(self):
        """Тест за сравнение на разширенията"""
        time_points = [1e9, 2e9, 3e9, 4e9, 5e9]
        
        comparison = self.calculator.compare_expansion_types(time_points)
        
        # Результатът трябва да съдържа данни за двете системи
        self.assertIn('abs_system', comparison)
        self.assertIn('rel_system', comparison)
        self.assertIn('linearity_difference', comparison)
        
        # АКС трябва да е по-линейна от РКС
        self.assertLess(comparison['abs_system']['linearity_measure'], 
                       comparison['rel_system']['linearity_measure'])

class TestCosmologyVisualizer(unittest.TestCase):
    """Тестове за визуализатора"""
    
    def setUp(self):
        """Подготовка за тестовете"""
        self.params = CosmologicalParameters()
        self.visualizer = CosmologyVisualizer(self.params)
    
    def test_visualizer_creation(self):
        """Тест за създаване на визуализатор"""
        self.assertIsInstance(self.visualizer.params, CosmologicalParameters)
        self.assertIsInstance(self.visualizer.calculator, ExpansionCalculator)

class TestIntegration(unittest.TestCase):
    """Интеграционни тестове"""
    
    def test_full_workflow(self):
        """Тест за пълен работен поток"""
        # Създаване на параметри
        params = CosmologicalParameters()
        
        # Създаване на АКС и РКС
        acs = AbsoluteCoordinateSystem(5e9, params)
        rcs = RelativeCoordinateSystem(10e9, params)
        
        # Създаване на калкулатор
        calculator = ExpansionCalculator(params)
        
        # Изчисляване на коефициенти
        abs_coeff = calculator.calculate_abs_expansion_coefficient(1e9, 2e9)
        rel_coeff = calculator.calculate_rel_expansion_coefficient(1e9, 2e9)
        
        # Всички стойности трябва да са разумни
        self.assertGreater(acs.scale_factor, 0)
        self.assertGreater(rcs.scale_factor, 0)
        self.assertGreater(abs_coeff, 0)
        self.assertGreater(rel_coeff, 0)
        
        # РКС трябва да има по-голям коефициент поради кубичното разширение
        self.assertGreater(rel_coeff, abs_coeff)
    
    def test_mathematical_consistency(self):
        """Тест за математическа консистентност"""
        params = CosmologicalParameters()
        
        # Времеви точки
        times = [1e9, 5e9, 10e9]
        
        # Проверка на мащабните фактори
        for t in times:
            acs = AbsoluteCoordinateSystem(t, params)
            rcs = RelativeCoordinateSystem(t, params)
            
            # Мащабните фактори трябва да са положителни
            self.assertGreater(acs.scale_factor, 0)
            self.assertGreater(rcs.scale_factor, 0)
            
            # Плътността трябва да е положителна
            self.assertGreater(acs.density, 0)
            
            # Темпът на време трябва да е положителен
            self.assertGreater(acs.time_rate, 0)
    
    def test_time_consistency(self):
        """Тест за консистентност на времето"""
        params = CosmologicalParameters()
        
        # Проверка че по-късните времена дават по-големи мащабни фактори
        times = [1e9, 5e9, 10e9, 13e9]
        
        abs_scale_factors = []
        rel_scale_factors = []
        
        for t in times:
            acs = AbsoluteCoordinateSystem(t, params)
            rcs = RelativeCoordinateSystem(t, params)
            
            abs_scale_factors.append(acs.scale_factor)
            rel_scale_factors.append(rcs.scale_factor)
        
        # Мащабните фактори трябва да нарастват монотонно
        for i in range(len(abs_scale_factors) - 1):
            self.assertGreater(abs_scale_factors[i+1], abs_scale_factors[i])
            self.assertGreater(rel_scale_factors[i+1], rel_scale_factors[i])

def run_performance_tests():
    """Стартира тестове за производителност"""
    import time
    
    print("=== ТЕСТОВЕ ЗА ПРОИЗВОДИТЕЛНОСТ ===")
    
    params = CosmologicalParameters()
    
    # Тест за създаване на АКС
    start_time = time.time()
    for i in range(1000):
        acs = AbsoluteCoordinateSystem(1e9 + i * 1e6, params)
    end_time = time.time()
    print(f"1000 АКС създадени за: {end_time - start_time:.4f} секунди")
    
    # Тест за изчисляване на коефициенти
    calculator = ExpansionCalculator(params)
    start_time = time.time()
    for i in range(1000):
        coeff = calculator.calculate_abs_expansion_coefficient(1e9, 2e9)
    end_time = time.time()
    print(f"1000 коефициента изчислени за: {end_time - start_time:.4f} секунди")
    
    # Тест за трансформация на координати
    rcs = RelativeCoordinateSystem(10e9, params)
    coords = np.array([1.0, 2.0, 3.0])
    start_time = time.time()
    for i in range(1000):
        transformed = rcs.transform_from_abs(coords, 5e9)
    end_time = time.time()
    print(f"1000 трансформации за: {end_time - start_time:.4f} секунди")

if __name__ == "__main__":
    print("СТАРТИРАНЕ НА ТЕСТОВЕТЕ ЗА НЕЛИНЕЙНО ВРЕМЕ КОСМОЛОГИЯ")
    print("=" * 60)
    
    # Стартиране на unit тестовете
    unittest.main(verbosity=2, exit=False)
    
    # Стартиране на тестовете за производителност
    run_performance_tests()
    
    print("\n" + "=" * 60)
    print("ВСИЧКИ ТЕСТОВЕ ЗАВЪРШЕНИ!") 