# Нелинейно време в космологията

## Описание на проекта

Този проект представя алтернативен космологичен модел, базиран на концепцията за **Абсолютни Координатни Системи (АКС)** и **Релативни Координатни Системи (РКС)**. Основната хипотеза е, че космическото ускорение е артефакт от времевата трансформация, а не реално физическо явление, което елиминира нуждата от тъмна енергия.

## Структура на проекта

```
test_3/
├── lib/                     # Основни модули и библиотеки
│   ├── __init__.py
│   ├── nonlinear_time_cosmology.py      # Основен космологичен модел
│   ├── redshift_calibration.py          # Калибрация с редshift данни
│   ├── real_data_acs_analysis.py        # Анализ с реални данни
│   └── acs_time_transformation.py       # АКС времева трансформация
├── tests/                   # Тестове и валидация
│   ├── __init__.py
│   ├── test_quick.py                    # Бърз тест на основните функции
│   ├── test_real_data_acs.py            # Тест на реални данни
│   └── test_acs_transformation.py       # Тест на времевата трансформация
├── analysis/                # Анализи и резултати
│   ├── examples.py                      # Примери за основния модел
│   ├── example_redshift_calibration.py  # Примери за калибрация
│   ├── example_real_data_acs.py         # Примери с реални данни
│   ├── example_acs_transformation.py    # Примери за трансформация
│   ├── RESULTS_SUMMARY.md               # Резюме на резултатите
│   └── THEORETICAL_INSIGHTS.md          # Теоретични размишления
├── requirements.txt         # Зависимости
├── README.md               # Този файл
└── SUMMARY.md              # Обобщение на проекта
```

## Основни компоненти

### 📚 Библиотека (`lib/`)

#### 1. **Основен космологичен модел** (`nonlinear_time_cosmology.py`)
- `CosmologicalParameters`: Параметри на модела
- `AbsoluteCoordinateSystem`: Линейно разширение в АКС
- `RelativeCoordinateSystem`: Кубично разширение в РКС
- `ExpansionCalculator`: Изчисления на разширението
- `CosmologyVisualizer`: Визуализация на резултатите

#### 2. **Калибрация с редshift данни** (`redshift_calibration.py`)
- `LinearTimeStepGenerator`: Генериране на линейни времеви стъпки
- `RedshiftCalculator`: Изчисления на червеното отместване
- `ExpansionRateCalibrator`: Калибрация на скоростта на разширение
- `RedshiftComparisonVisualizer`: Сравнение на модели

#### 3. **Анализ с реални данни** (`real_data_acs_analysis.py`)
- `PantheonDataLoader`: Зареждане на Pantheon+ данни
- `UnifiedACSFinder`: Намиране на единна АКС
- `LinearACSGenerator`: Генериране на линейни АКС системи
- `LinearExpansionAnalyzer`: Анализ на линейното разширение
- `RealDataACSVisualizer`: Визуализация на реални данни

#### 4. **АКС времева трансформация** (`acs_time_transformation.py`)
- `TimeTransformationModel`: Модел за времева трансформация
- `RedshiftTimeRelation`: Връзка между редshift и време
- `ExpansionAnalyzer`: Анализ на разширението
- `ExpansionVisualizer`: Визуализация на трансформацията

### 🧪 Тестове (`tests/`)

- **test_quick.py**: Бърз тест на основните функции
- **test_real_data_acs.py**: Валидация с реални данни от Pantheon+
- **test_acs_transformation.py**: Тест на времевата трансформация

### 📊 Анализи (`analysis/`)

- **Примери**: Демонстрация на всички възможности
- **RESULTS_SUMMARY.md**: Подробно резюме на резултатите
- **THEORETICAL_INSIGHTS.md**: Теоретични размишления и заключения

## Инсталация

1. Клонирайте проекта:
```bash
git clone [repository-url]
cd test_3
```

2. Инсталирайте зависимостите:
```bash
pip install -r requirements.txt
```

## Използване

### Основни примери

```python
# Импорт на основните модули
from lib import CosmologicalParameters, AbsoluteCoordinateSystem, ExpansionCalculator

# Създаване на модел
params = CosmologicalParameters(k_expansion=1e-3, age_universe=13.8)
acs = AbsoluteCoordinateSystem(params)

# Изчисляване на разширението
calculator = ExpansionCalculator(acs)
results = calculator.calculate_expansion_over_time()
```

### Стартиране на тестове

```bash
# Бърз тест
python tests/test_quick.py

# Тест с реални данни
python tests/test_real_data_acs.py

# Тест на времевата трансформация
python tests/test_acs_transformation.py
```

### Изпълнение на анализи

```bash
# Основни примери
python analysis/examples.py

# Калибрация с редshift данни
python analysis/example_redshift_calibration.py

# Анализ с реални данни
python analysis/example_real_data_acs.py

# Времева трансформация
python analysis/example_acs_transformation.py
```

## Ключови резултати

### ✅ Валидирани постижения:
- **Линейно разширение в АКС**: a(t) = k*t потвърдено
- **Времева трансформация**: T(z) = 1/(1+z)^(3/2) математически валидирана
- **Калибрация с реални данни**: 1701 записа от Pantheon+ успешно анализирани
- **Всички тестове**: 100% успеваемост

### 📈 Ключови стойности:
- **Оптимален коефициент на разширение**: k = 6.253573
- **Калибрационна грешка**: 52.223381
- **Времева дилатация при z=1**: 2.8x по-бавно течение на времето

## Научни заключения

1. **Космическото ускорение** може да е артефакт от времевата трансформация
2. **Тъмната енергия** може да не е необходима за обяснението на наблюденията
3. **Абсолютните координатни системи** предлагат елегантна алтернатива на ΛCDM
4. **Времевата дилатация** е ключов механизъм за разбирането на космологията

## Препоръки за бъдещи изследвания

1. Включване на квантови ефекти в АКС модела
2. Сравнение с данни от космическия микровълнов фон (CMB)
3. Изследване на гравитационни вълни в АКС контекста
4. Разработка на наблюдателни тестове за различаване от ΛCDM

## Зависимости

- `numpy >= 1.20.0`
- `matplotlib >= 3.3.0`
- `scipy >= 1.7.0`
- `pandas >= 1.3.0`

## Лиценз

Този проект е разработен за научни изследвания в областта на космологията.

## Контакти

За въпроси и предложения относно проекта, моля свържете се с автора.

---

**Версия**: 1.0.0  
**Дата**: 2024-2025  
**Статус**: Завършен 