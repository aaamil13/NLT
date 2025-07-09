#!/usr/bin/env python3
"""
Модул за оптимизационни алгоритми
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, basinhopping, minimize
from scipy.stats import multivariate_normal
import multiprocessing as mp
from typing import Callable, Dict, List, Tuple, Any, Optional
import time
import warnings

# Потискаме предупреждения
warnings.filterwarnings('ignore')

# Глобална тестова функция
def rosenbrock_global(x):
    """Глобална Rosenbrock функция"""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


class DifferentialEvolutionOptimizer:
    """
    Клас за оптимизация чрез Differential Evolution
    """
    
    def __init__(self, 
                 population_size: int = 15,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 mutation_factor: float = 0.5,
                 crossover_probability: float = 0.7,
                 seed: Optional[int] = None,
                 parallel: bool = True):
        """
        Инициализация на оптимизатора
        
        Args:
            population_size: Размер на популацията
            max_iterations: Максимален брой итерации
            tolerance: Толеранс за конвергенция
            mutation_factor: Фактор на мутация
            crossover_probability: Вероятност за кроссовър
            seed: Seed за възпроизводимост
            parallel: Дали да използва паралелизация
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.mutation_factor = mutation_factor
        self.crossover_probability = crossover_probability
        self.seed = seed
        self.parallel = parallel
        self.history = []
        self.best_params = None
        self.best_score = np.inf
        
    def optimize(self, 
                 objective_function: Callable,
                 bounds: List[Tuple[float, float]],
                 args: Tuple = (),
                 callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Оптимизация на целевата функция
        
        Args:
            objective_function: Функция за оптимизация
            bounds: Граници на параметрите
            args: Допълнителни аргументи за функцията
            callback: Callback функция за мониторинг
            
        Returns:
            Резултати от оптимизацията
        """
        start_time = time.time()
        
        # Callback за записване на историята
        def internal_callback(xk, convergence):
            score = objective_function(xk, *args)
            self.history.append({
                'iteration': len(self.history),
                'parameters': xk.copy(),
                'score': score,
                'convergence': convergence
            })
            
            if score < self.best_score:
                self.best_score = score
                self.best_params = xk.copy()
            
            if callback:
                callback(xk, score, convergence)
            
            return False  # Не спираме оптимизацията
        
        # Настройваме параметри за DE
        workers = mp.cpu_count() if self.parallel else 1
        
        result = differential_evolution(
            objective_function,
            bounds,
            args=args,
            strategy='best1bin',
            maxiter=self.max_iterations,
            popsize=self.population_size,
            tol=self.tolerance,
            mutation=self.mutation_factor,
            recombination=self.crossover_probability,
            seed=self.seed,
            callback=internal_callback,
            workers=workers,
            updating='deferred' if self.parallel else 'immediate'
        )
        
        end_time = time.time()
        
        return {
            'success': result.success,
            'best_parameters': result.x,
            'best_score': result.fun,
            'iterations': result.nit,
            'evaluations': result.nfev,
            'message': result.message,
            'execution_time': end_time - start_time,
            'history': self.history
        }
    
    def plot_convergence(self, save_path: str = None):
        """
        Графика на конвергенцията
        
        Args:
            save_path: Път за записване на графиката
        """
        if not self.history:
            print("Няма данни за конвергенция")
            return
        
        iterations = [h['iteration'] for h in self.history]
        scores = [h['score'] for h in self.history]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(iterations, scores, 'b-', linewidth=2)
        plt.xlabel('Итерация')
        plt.ylabel('Стойност на целевата функция')
        plt.title('Конвергенция на Differential Evolution')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        if len(self.best_params) <= 6:  # Показваме само първите 6 параметъра
            param_history = np.array([h['parameters'] for h in self.history])
            for i in range(len(self.best_params)):
                plt.plot(iterations, param_history[:, i], label=f'Параметър {i+1}')
            plt.xlabel('Итерация')
            plt.ylabel('Стойност на параметъра')
            plt.title('Еволюция на параметрите')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class BasinhoppingOptimizer:
    """
    Клас за оптимизация чрез Basinhopping (глобална оптимизация)
    """
    
    def __init__(self,
                 n_iterations: int = 100,
                 temperature: float = 1.0,
                 step_size: float = 0.5,
                 interval: int = 50,
                 local_optimizer: str = 'L-BFGS-B',
                 seed: Optional[int] = None):
        """
        Инициализация на basinhopping оптимизатора
        
        Args:
            n_iterations: Брой итерации
            temperature: Температура за Metropolis критерий
            step_size: Размер на стъпката
            interval: Интервал за адаптивно настройване
            local_optimizer: Локален оптимизатор
            seed: Seed за възпроизводимост
        """
        self.n_iterations = n_iterations
        self.temperature = temperature
        self.step_size = step_size
        self.interval = interval
        self.local_optimizer = local_optimizer
        self.seed = seed
        self.history = []
        self.best_params = None
        self.best_score = np.inf
        
    def optimize(self,
                 objective_function: Callable,
                 initial_guess: np.ndarray,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 args: Tuple = (),
                 callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Оптимизация с basinhopping
        
        Args:
            objective_function: Функция за оптимизация
            initial_guess: Начална оценка
            bounds: Граници на параметрите
            args: Допълнителни аргументи
            callback: Callback функция
            
        Returns:
            Резултати от оптимизацията
        """
        start_time = time.time()
        
        # Callback за записване на историята
        def internal_callback(x, f, accept):
            self.history.append({
                'iteration': len(self.history),
                'parameters': x.copy(),
                'score': f,
                'accepted': accept
            })
            
            if f < self.best_score:
                self.best_score = f
                self.best_params = x.copy()
            
            if callback:
                callback(x, f, accept)
            
            return False  # Не спираме оптимизацията
        
        # Настройваме локалния оптимизатор
        minimizer_kwargs = {'method': self.local_optimizer}
        if bounds:
            minimizer_kwargs['bounds'] = bounds
        if args:
            minimizer_kwargs['args'] = args
        
        # Настройваме seed за възпроизводимост
        if self.seed:
            np.random.seed(self.seed)
        
        result = basinhopping(
            objective_function,
            initial_guess,
            niter=self.n_iterations,
            T=self.temperature,
            stepsize=self.step_size,
            interval=self.interval,
            minimizer_kwargs=minimizer_kwargs,
            callback=internal_callback,
            seed=self.seed
        )
        
        end_time = time.time()
        
        return {
            'success': result.nit < self.n_iterations,  # Проверка за успех
            'best_parameters': result.x,
            'best_score': result.fun,
            'iterations': result.nit,
            'evaluations': result.nfev,
            'message': result.message if hasattr(result, 'message') else 'Завършено',
            'execution_time': end_time - start_time,
            'history': self.history
        }
    
    def plot_convergence(self, save_path: str = None):
        """
        Графика на конвергенцията за basinhopping
        
        Args:
            save_path: Път за записване на графиката
        """
        if not self.history:
            print("Няма данни за конвергенция")
            return
        
        iterations = [h['iteration'] for h in self.history]
        scores = [h['score'] for h in self.history]
        accepted = [h['accepted'] for h in self.history]
        
        plt.figure(figsize=(15, 10))
        
        # График 1: Конвергенция
        plt.subplot(2, 2, 1)
        plt.plot(iterations, scores, 'b-', linewidth=1, alpha=0.7)
        # Добавяме линия за най-добрите стойности
        best_so_far = []
        current_best = np.inf
        for score in scores:
            if score < current_best:
                current_best = score
            best_so_far.append(current_best)
        plt.plot(iterations, best_so_far, 'r-', linewidth=2, label='Най-добра стойност')
        plt.xlabel('Итерация')
        plt.ylabel('Стойност на целевата функция')
        plt.title('Конвергенция на Basinhopping')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # График 2: Приемане на стъпки
        plt.subplot(2, 2, 2)
        accept_rate = np.cumsum(accepted) / np.arange(1, len(accepted) + 1)
        plt.plot(iterations, accept_rate, 'g-', linewidth=2)
        plt.xlabel('Итерация')
        plt.ylabel('Процент приемане')
        plt.title('Степен на приемане на стъпки')
        plt.grid(True, alpha=0.3)
        
        # График 3: Хистограма на стойностите
        plt.subplot(2, 2, 3)
        plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Стойност на целевата функция')
        plt.ylabel('Честота')
        plt.title('Разпределение на стойностите')
        plt.grid(True, alpha=0.3)
        
        # График 4: Еволюция на параметрите (само първите 4)
        plt.subplot(2, 2, 4)
        if len(self.best_params) <= 4:
            param_history = np.array([h['parameters'] for h in self.history])
            for i in range(min(len(self.best_params), 4)):
                plt.plot(iterations, param_history[:, i], label=f'Параметър {i+1}')
            plt.xlabel('Итерация')
            plt.ylabel('Стойност на параметъра')
            plt.title('Еволюция на параметрите')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class HybridOptimizer:
    """
    Хибриден оптимизатор, който комбинира различни методи
    """
    
    def __init__(self,
                 methods: List[str] = ['differential_evolution', 'basinhopping'],
                 de_params: Dict = None,
                 bh_params: Dict = None):
        """
        Инициализация на хибридния оптимизатор
        
        Args:
            methods: Списък с методи за използване
            de_params: Параметри за Differential Evolution
            bh_params: Параметри за Basinhopping
        """
        self.methods = methods
        self.de_params = de_params or {}
        self.bh_params = bh_params or {}
        self.results = {}
        
    def optimize(self,
                 objective_function: Callable,
                 bounds: List[Tuple[float, float]],
                 args: Tuple = ()) -> Dict[str, Any]:
        """
        Оптимизация с множество методи
        
        Args:
            objective_function: Функция за оптимизация
            bounds: Граници на параметрите
            args: Допълнителни аргументи
            
        Returns:
            Резултати от всички методи
        """
        all_results = {}
        
        for method in self.methods:
            print(f"Стартиране на {method}...")
            
            if method == 'differential_evolution':
                optimizer = DifferentialEvolutionOptimizer(**self.de_params)
                result = optimizer.optimize(objective_function, bounds, args)
                all_results[method] = result
                
            elif method == 'basinhopping':
                # За basinhopping трябва начална точка
                initial_guess = np.array([(b[0] + b[1]) / 2 for b in bounds])
                optimizer = BasinhoppingOptimizer(**self.bh_params)
                result = optimizer.optimize(objective_function, initial_guess, bounds, args)
                all_results[method] = result
        
        # Намираме най-добрия резултат
        best_method = min(all_results.keys(), 
                         key=lambda x: all_results[x]['best_score'])
        
        return {
            'best_method': best_method,
            'best_result': all_results[best_method],
            'all_results': all_results
        }
    
    def compare_methods(self, save_path: str = None):
        """
        Сравнение на методите
        
        Args:
            save_path: Път за записване на графиката
        """
        if not self.results:
            print("Няма резултати за сравнение")
            return
        
        methods = list(self.results['all_results'].keys())
        scores = [self.results['all_results'][m]['best_score'] for m in methods]
        times = [self.results['all_results'][m]['execution_time'] for m in methods]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(methods, scores)
        plt.xlabel('Метод')
        plt.ylabel('Най-добра стойност')
        plt.title('Сравнение на качеството')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        plt.bar(methods, times)
        plt.xlabel('Метод')
        plt.ylabel('Време за изпълнение [s]')
        plt.title('Сравнение на скоростта')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Помощни функции
def create_bounds(center: List[float], 
                 widths: List[float]) -> List[Tuple[float, float]]:
    """
    Създава граници около централна точка
    
    Args:
        center: Централни стойности
        widths: Ширина на интервалите
        
    Returns:
        Списък с граници
    """
    bounds = []
    for c, w in zip(center, widths):
        bounds.append((c - w/2, c + w/2))
    return bounds


def test_optimization_methods():
    """
    Тестова функция за оптимизационните методи
    """
    # Граници
    bounds = [(-5, 5), (-5, 5)]
    
    # Тестваме DE
    print("Тестване на Differential Evolution...")
    de_opt = DifferentialEvolutionOptimizer(max_iterations=100)
    de_result = de_opt.optimize(rosenbrock_global, bounds)
    print(f"DE резултат: {de_result['best_score']:.6f}")
    
    # Тестваме Basinhopping
    print("Тестване на Basinhopping...")
    bh_opt = BasinhoppingOptimizer(n_iterations=100)
    initial_guess = np.array([0.0, 0.0])
    bh_result = bh_opt.optimize(rosenbrock_global, initial_guess, bounds)
    print(f"BH резултат: {bh_result['best_score']:.6f}")
    
    # Тестваме хибридния оптимизатор
    print("Тестване на хибридния оптимизатор...")
    hybrid_opt = HybridOptimizer()
    hybrid_result = hybrid_opt.optimize(rosenbrock_global, bounds)
    print(f"Най-добър метод: {hybrid_result['best_method']}")
    print(f"Най-добър резултат: {hybrid_result['best_result']['best_score']:.6f}")


if __name__ == "__main__":
    test_optimization_methods() 