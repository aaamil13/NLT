# Валидационна система за теория на нелинейно време

## Описание

Тази валидационна система предоставя обширен набор от тестове за валидиране на теорията за нелинейно време и абсолютна координатна система (АКС). Системата включва:

- **GPS тестове** - Анализ на GPS данни за времева дилатация
- **Анализ на остатъчен шум** - Детайлен анализ на остатъците от моделите
- **Първобитен анализ** - Анализ на рекомбинацията, остатъчния шум и първобитните флуктуации
- **Статистическа значимост** - Тестове за статистическата значимост
- **Оптимизационни методи** - Differential Evolution, Basinhopping, хибридни методи
- **MCMC и Байесов анализ** - Модел селекция и параметрично оценяване
- **Обработка на сурови данни** - Работа с SH0ES и Pantheon+ данни БЕЗ ΛCDM адаптации

## Структура

```
validation_tests/
├── gps_tests/                    # GPS тестове
│   ├── gps_time_dilation.py      # Основен GPS тест
│   ├── gps_orbit_analysis.py     # Орбитален анализ
│   ├── gps_precision_tests.py    # Тестове за прецизност
│   └── gps_relativistic_effects.py # Релативистични ефекти
├── residual_noise_tests/         # Тестове за остатъчен шум
│   └── residual_noise_analyzer.py # Обширен анализатор
├── primordial_analysis/          # Първобитен анализ
│   ├── recombination_analysis.py # Анализ на рекомбинацията
│   ├── relic_noise_analyzer.py   # Анализ на остатъчния шум
│   └── primordial_fluctuations.py # Първобитни флуктуации
├── statistical_significance_tests/ # Статистически тестове
├── optimization_methods/         # Оптимизационни методи
├── raw_data_processing/          # Обработка на сурови данни
├── common_utils/                 # Общи утилити
│   ├── optimization_engines.py   # Оптимизационни двигатели
│   ├── mcmc_bayesian.py          # MCMC и Байесов анализ
│   ├── statistical_tests.py     # Статистически тестове
│   └── data_processors.py       # Процесори за данни
└── run_comprehensive_validation.py # Основен стартов скрипт
```

## Използване

### Бързо стартиране

```bash
cd validation_tests
python run_comprehensive_validation.py
```

### Демонстрация на първобитния анализ

```bash
cd validation_tests
python demo_primordial_analysis.py
```

Този скрипт демонстрира:
- Анализ на рекомбинацията с удължен период
- Анализ на остатъчния шум от създаването на Вселената  
- Комплексен анализ на първобитните флуктуации
- Теоретичните импликации за ранно структурообразуване

Системата ще ви предложи три опции:
1. **Бърз тест** - Основни компоненти (GPS + остатъчен шум)
2. **Пълен тест** - Всички компоненти
3. **Персонализиран тест** - Избирате какви тестове да стартирате

### Програмно използване

```python
from validation_tests.run_comprehensive_validation import ComprehensiveValidationSuite

# Създаване на валидационна система
suite = ComprehensiveValidationSuite()

# Стартиране на бърз тест
results = suite.run_quick_test()

# Стартиране на пълен тест
results = suite.run_comprehensive_test()

# Персонализиран тест
results = suite.run_all_tests(
    include_gps=True,
    include_residual=True,
    include_primordial=True,
    include_data_processing=False,
    include_optimization=True,
    include_mcmc=False,
    include_statistical=True
)
```

## Отделни модули

### GPS тестове

```python
from validation_tests.gps_tests.gps_time_dilation import GPSTimeDilationTest

# Тест с нелинейно време
test = GPSTimeDilationTest(use_nonlinear_time=True)
results = test.run_comprehensive_test()
```

### Анализ на остатъчен шум

```python
from validation_tests.residual_noise_tests.residual_noise_analyzer import ResidualNoiseAnalyzer

# Анализатор с реални данни
analyzer = ResidualNoiseAnalyzer(use_raw_data=True)
results = analyzer.run_comprehensive_analysis()
```

### Първобитен анализ

```python
from validation_tests.primordial_analysis import (
    RecombinationAnalyzer, 
    RelicNoiseAnalyzer,
    PrimordialFluctuationAnalyzer
)

# Анализ на рекомбинацията
recomb_analyzer = RecombinationAnalyzer(z_recomb=1100)
tau = np.linspace(1.0, 4.0, 1000)
rho = recomb_analyzer.energy_density_evolution(tau)
analysis = recomb_analyzer.plot_recombination_analysis(tau, rho)

# Анализ на остатъчния шум от създаването на Вселената
relic_analyzer = RelicNoiseAnalyzer(cmb_amplitude=1e-5)
delta_rho = relic_analyzer.generate_primordial_noise(tau, spectral_index=0.96)
spectral_analysis = relic_analyzer.plot_comprehensive_analysis(tau, rho, delta_rho)

# Пълен анализ на първобитните флуктуации
primordial_analyzer = PrimordialFluctuationAnalyzer()
results = primordial_analyzer.run_complete_analysis()
```

### Оптимизационни методи

```python
from validation_tests.common_utils.optimization_engines import DifferentialEvolutionOptimizer, BasinhoppingOptimizer

# Differential Evolution
de_optimizer = DifferentialEvolutionOptimizer(max_iterations=1000)
result = de_optimizer.optimize(objective_function, bounds)

# Basinhopping
bh_optimizer = BasinhoppingOptimizer(n_iterations=500)
result = bh_optimizer.optimize(objective_function, initial_guess, bounds)
```

### MCMC и Байесов анализ

```python
from validation_tests.common_utils.mcmc_bayesian import MCMCBayesianAnalyzer, BayesianModelComparison

# MCMC анализ
analyzer = MCMCBayesianAnalyzer(n_walkers=50, n_steps=1000)
result = analyzer.run_mcmc(log_probability, initial_params, bounds)

# Байесово сравнение на модели
comparison = BayesianModelComparison()
comparison.add_model('model1', log_likelihood1, log_prior1, bounds1, initial1)
comparison.add_model('model2', log_likelihood2, log_prior2, bounds2, initial2)
results = comparison.run_comparison(data)
```

### Статистически тестове

```python
from validation_tests.common_utils.statistical_tests import StatisticalSignificanceTest

# Статистически тестове
stat_test = StatisticalSignificanceTest()
analysis = stat_test.comprehensive_residual_analysis(residuals, fitted_values)
```

### Обработка на сурови данни

```python
from validation_tests.common_utils.data_processors import RawDataProcessor

# Обработка на данни
processor = RawDataProcessor()
pantheon_data = processor.load_pantheon_plus_data()
shoes_data = processor.load_shoes_data()
unified_data = processor.extract_redshift_magnitude_data()
```

## Изисквания

### Основни зависимости

```python
numpy
matplotlib
scipy
pandas
emcee
corner
astropy
sklearn
```

### Опционални зависимости

```python
statsmodels  # За по-добър автокорелационен анализ
```

## Данни

Системата работи с:

- **Pantheon+ данни** - Трябва да са в `../test_2/data/Pantheon+_Data/`
- **SH0ES данни** - Трябва да са в `../test_2/data/SH0ES_Data/`
- **Синтетични данни** - Генерират се автоматично ако няма реални данни

## Резултати

Системата генерира:

- **Графики** - Автоматично се показват по време на изпълнение
- **Доклади** - Текстови доклади с резултати
- **Статистики** - Детайлни статистички анализи
- **Сравнения** - Сравнения между модели

## Ключови характеристики

### Оптимизационни методи

- **Differential Evolution** - Глобална оптимизация с еволюционни алгоритми
- **Basinhopping** - Комбинация от локална и глобална оптимизация
- **Хибридни методи** - Комбинация от различни техники
- **Паралелизация** - Използване на множество CPU ядра

### Статистически анализи

- **Нормалност** - Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling
- **Автокорелация** - Runs test, Durbin-Watson
- **Outliers** - IQR и Z-score методи

### Първобитен анализ

- **Рекомбинация** - Анализ на удълженият рекомбинационен период в нелинейно време
- **Остатъчен шум** - Спектрален анализ на първобитните квантови флуктуации
- **Структурообразуване** - Анализ на ранно галактическо формиране
- **Джийнс нестабилност** - Критерии за гравитационно колапсиране
- **Сравнение с ΛCDM** - Статистическо сравнение със стандартната космология
- **Спектрален анализ** - FFT, степенни спектри, спектрална ентропия

### MCMC и Байесови методи

- **Ensemble sampler** - Използване на emcee библиотеката
- **Информационни критерии** - AIC, BIC, DIC, WAIC
- **Модел селекция** - Автоматично сравнение на модели
- **Posterior анализ** - Corner plots, доверителни интервали

## Примерни резултати

Системата показва:

- Съвместимост на GPS данните с теорията за нелинейно време
- Статистическа значимост на модела спрямо класическите алтернативи
- Качество на остатъчния шум и неговата структура
- Байесово сравнение показващо предпочитание към нелинейното време

## Препоръки

1. **Започнете с бърз тест** за да се запознаете със системата
2. **Използвайте реални данни** когато е възможно
3. **Анализирайте резултатите** внимателно - системата дава много информация
4. **Комбинирайте различни методи** за по-надеждни резултати
5. **Запазвайте резултатите** за бъдещи сравнения

## Поддръжка

Системата е проектирана да бъде:
- **Модулна** - Всеки компонент може да се използва независимо
- **Разширима** - Лесно добавяне на нови тестове
- **Надеждна** - Обработка на грешки и fallback опции
- **Документирана** - Подробна документация във всеки модул

За въпроси и проблеми, проверете документацията в отделните модули. 