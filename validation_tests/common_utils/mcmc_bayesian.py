"""
MCMC и Байесов анализ за модел селекция
======================================

Този модул имплементира:
- Markov Chain Monte Carlo (MCMC) сэмплинг
- Байесов анализ за модел селекция
- Информационни критерии (AIC, BIC, DIC)
- Posterior разпределения
- Модел сравнение

Автор: Система за анализ на нелинейно време
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm, gamma, uniform
from scipy.optimize import minimize
import emcee
import corner
from typing import Callable, Dict, List, Tuple, Any, Optional
import time
import warnings

warnings.filterwarnings('ignore')


class MCMCBayesianAnalyzer:
    """
    Клас за MCMC и Байесов анализ
    """
    
    def __init__(self,
                 n_walkers: int = 50,
                 n_steps: int = 1000,
                 n_burn: int = 200,
                 thin: int = 1,
                 seed: Optional[int] = None):
        """
        Инициализация на MCMC анализатора
        
        Args:
            n_walkers: Брой ходачи (walkers)
            n_steps: Брой стъпки за всеки ходач
            n_burn: Брой стъпки за отхвърляне (burn-in)
            thin: Разредяване на веригата
            seed: Seed за възпроизводимост
        """
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_burn = n_burn
        self.thin = thin
        self.seed = seed
        
        self.sampler = None
        self.samples = None
        self.log_prob = None
        self.acceptance_rate = None
        
    def run_mcmc(self,
                 log_probability: Callable,
                 initial_params: np.ndarray,
                 param_bounds: List[Tuple[float, float]],
                 args: Tuple = (),
                 progress: bool = True) -> Dict[str, Any]:
        """
        Стартира MCMC сэмплинг
        
        Args:
            log_probability: Функция за логаритъм на вероятността
            initial_params: Начални параметри
            param_bounds: Граници на параметрите
            args: Допълнителни аргументи
            progress: Дали да показва прогрес
            
        Returns:
            Резултати от сэмплинга
        """
        start_time = time.time()
        
        # Настройваме seed
        if self.seed:
            np.random.seed(self.seed)
        
        # Брой параметри
        n_params = len(initial_params)
        
        # Създаваме начални позиции за всички ходачи
        pos = self._initialize_walkers(initial_params, param_bounds)
        
        # Wrapper функция за log_probability с граничните условия
        def log_prob_wrapper(params):
            # Проверяваме дали параметрите са в границите
            for i, (low, high) in enumerate(param_bounds):
                if params[i] < low or params[i] > high:
                    return -np.inf
            
            return log_probability(params, *args)
        
        # Създаваме sampler
        self.sampler = emcee.EnsembleSampler(
            self.n_walkers,
            n_params,
            log_prob_wrapper
        )
        
        # Стартираме сэмплинга
        self.sampler.run_mcmc(pos, self.n_steps, progress=progress)
        
        # Извличаме резултатите
        self.samples = self.sampler.get_chain(discard=self.n_burn, thin=self.thin, flat=True)
        self.log_prob = self.sampler.get_log_prob(discard=self.n_burn, thin=self.thin, flat=True)
        self.acceptance_rate = np.mean(self.sampler.acceptance_fraction)
        
        end_time = time.time()
        
        # Анализ на резултатите
        analysis = self._analyze_results()
        
        return {
            'samples': self.samples,
            'log_prob': self.log_prob,
            'acceptance_rate': self.acceptance_rate,
            'execution_time': end_time - start_time,
            'analysis': analysis
        }
    
    def _initialize_walkers(self,
                           initial_params: np.ndarray,
                           param_bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Инициализира позициите на ходачите
        
        Args:
            initial_params: Начални параметри
            param_bounds: Граници на параметрите
            
        Returns:
            Начални позиции за всички ходачи
        """
        n_params = len(initial_params)
        pos = np.zeros((self.n_walkers, n_params))
        
        for i in range(self.n_walkers):
            for j, (low, high) in enumerate(param_bounds):
                # Инициализираме около началната стойност с малка дисперсия
                sigma = (high - low) * 0.01  # 1% от диапазона
                pos[i, j] = np.random.normal(initial_params[j], sigma)
                
                # Проверяваме границите
                pos[i, j] = np.clip(pos[i, j], low, high)
        
        return pos
    
    def _analyze_results(self) -> Dict[str, Any]:
        """
        Анализира резултатите от MCMC
        
        Returns:
            Статистики на резултатите
        """
        if self.samples is None:
            return {}
        
        # Основни статистики
        mean_params = np.mean(self.samples, axis=0)
        median_params = np.median(self.samples, axis=0)
        std_params = np.std(self.samples, axis=0)
        
        # Процентили
        percentiles = np.percentile(self.samples, [16, 50, 84], axis=0)
        
        # Effective sample size
        try:
            autocorr_time = self.sampler.get_autocorr_time()
            eff_sample_size = self.samples.shape[0] / (2 * autocorr_time)
        except:
            autocorr_time = np.full(self.samples.shape[1], np.nan)
            eff_sample_size = np.full(self.samples.shape[1], np.nan)
        
        # Gelman-Rubin статистика (R-hat)
        try:
            chain = self.sampler.get_chain(discard=self.n_burn, thin=self.thin)
            r_hat = self._calculate_r_hat(chain)
        except:
            r_hat = np.full(self.samples.shape[1], np.nan)
        
        return {
            'mean_params': mean_params,
            'median_params': median_params,
            'std_params': std_params,
            'percentiles_16_50_84': percentiles,
            'autocorr_time': autocorr_time,
            'eff_sample_size': eff_sample_size,
            'r_hat': r_hat,
            'n_effective_samples': len(self.samples)
        }
    
    def _calculate_r_hat(self, chain: np.ndarray) -> np.ndarray:
        """
        Пресмята Gelman-Rubin статистиката (R-hat)
        
        Args:
            chain: MCMC верига (n_steps, n_walkers, n_params)
            
        Returns:
            R-hat стойности за всеки параметър
        """
        n_steps, n_walkers, n_params = chain.shape
        
        r_hat = np.zeros(n_params)
        
        for i in range(n_params):
            # Извличаме данни за параметъра
            param_chains = chain[:, :, i]  # (n_steps, n_walkers)
            
            # Пресмятаме между- и вътре-верижната дисперсия
            chain_means = np.mean(param_chains, axis=0)
            global_mean = np.mean(chain_means)
            
            # Между-верижна дисперсия
            B = n_steps / (n_walkers - 1) * np.sum((chain_means - global_mean)**2)
            
            # Вътре-верижна дисперсия
            chain_vars = np.var(param_chains, axis=0, ddof=1)
            W = np.mean(chain_vars)
            
            # R-hat
            var_plus = (n_steps - 1) / n_steps * W + B / n_steps
            r_hat[i] = np.sqrt(var_plus / W)
        
        return r_hat
    
    def plot_chains(self, param_names: List[str] = None, save_path: str = None):
        """
        Графики на веригите
        
        Args:
            param_names: Имена на параметрите
            save_path: Път за записване
        """
        if self.sampler is None:
            print("Няма данни за верригите")
            return
        
        # Извличаме веригите
        samples = self.sampler.get_chain()
        
        n_params = samples.shape[2]
        if param_names is None:
            param_names = [f'Параметър {i+1}' for i in range(n_params)]
        
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params))
        if n_params == 1:
            axes = [axes]
        
        for i in range(n_params):
            ax = axes[i]
            
            # Показваме всички вериги
            for j in range(self.n_walkers):
                ax.plot(samples[:, j, i], alpha=0.3, color='blue')
            
            # Маркираме burn-in периода
            ax.axvline(x=self.n_burn, color='red', linestyle='--', alpha=0.7, label='Burn-in')
            
            ax.set_xlabel('Стъпка')
            ax.set_ylabel(param_names[i])
            ax.set_title(f'Верига за {param_names[i]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_corner(self, param_names: List[str] = None, save_path: str = None):
        """
        Corner plot на posterior разпределенията
        
        Args:
            param_names: Имена на параметрите
            save_path: Път за записване
        """
        if self.samples is None:
            print("Няма данни за corner plot")
            return
        
        n_params = self.samples.shape[1]
        if param_names is None:
            param_names = [f'Параметър {i+1}' for i in range(n_params)]
        
        # Създаваме corner plot
        fig = corner.corner(
            self.samples,
            labels=param_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={'fontsize': 12},
            label_kwargs={'fontsize': 14}
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_best_fit_params(self) -> Dict[str, Any]:
        """
        Извлича най-добрите параметри
        
        Returns:
            Най-добри параметри и техните статистики
        """
        if self.samples is None:
            return {}
        
        # Намираме индекса с най-висока вероятност
        best_idx = np.argmax(self.log_prob)
        best_params = self.samples[best_idx]
        
        # Медиани и доверителни интервали
        medians = np.median(self.samples, axis=0)
        percentiles = np.percentile(self.samples, [16, 84], axis=0)
        uncertainties = np.diff(percentiles, axis=0).flatten() / 2
        
        return {
            'best_fit_params': best_params,
            'best_log_prob': self.log_prob[best_idx],
            'median_params': medians,
            'uncertainties': uncertainties,
            'confidence_intervals': percentiles
        }


class BayesianModelComparison:
    """
    Клас за Байесово сравнение на модели
    """
    
    def __init__(self):
        """Инициализация на класа за модел сравнение"""
        self.models = {}
        self.results = {}
        
    def add_model(self,
                  model_name: str,
                  log_likelihood: Callable,
                  log_prior: Callable,
                  param_bounds: List[Tuple[float, float]],
                  initial_params: np.ndarray):
        """
        Добавя модел за сравнение
        
        Args:
            model_name: Име на модела
            log_likelihood: Функция за log-likelihood
            log_prior: Функция за log-prior
            param_bounds: Граници на параметрите
            initial_params: Начални параметри
        """
        self.models[model_name] = {
            'log_likelihood': log_likelihood,
            'log_prior': log_prior,
            'param_bounds': param_bounds,
            'initial_params': initial_params
        }
    
    def run_comparison(self,
                       data: Any,
                       mcmc_params: Dict = None) -> Dict[str, Any]:
        """
        Стартира сравнението на модели
        
        Args:
            data: Данни за анализ
            mcmc_params: Параметри за MCMC
            
        Returns:
            Резултати от сравнението
        """
        if mcmc_params is None:
            mcmc_params = {}
        
        model_results = {}
        
        for model_name, model in self.models.items():
            print(f"Анализ на модел: {model_name}")
            
            # Дефинираме log_probability
            def log_probability(params):
                lp = model['log_prior'](params)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + model['log_likelihood'](params, data)
            
            # Стартираме MCMC
            analyzer = MCMCBayesianAnalyzer(**mcmc_params)
            result = analyzer.run_mcmc(
                log_probability,
                model['initial_params'],
                model['param_bounds']
            )
            
            # Пресмятаме информационни критерии
            info_criteria = self._calculate_information_criteria(
                result['samples'],
                result['log_prob'],
                data
            )
            
            model_results[model_name] = {
                'mcmc_result': result,
                'information_criteria': info_criteria,
                'analyzer': analyzer
            }
        
        # Сравняваме модели
        comparison = self._compare_models(model_results)
        
        self.results = {
            'model_results': model_results,
            'comparison': comparison
        }
        
        return self.results
    
    def _calculate_information_criteria(self,
                                       samples: np.ndarray,
                                       log_prob: np.ndarray,
                                       data: Any) -> Dict[str, float]:
        """
        Пресмята информационни критерии
        
        Args:
            samples: MCMC семпли
            log_prob: Log-probability стойности
            data: Данни
            
        Returns:
            Информационни критерии
        """
        n_data = len(data) if hasattr(data, '__len__') else 1
        n_params = samples.shape[1]
        
        # Най-висока вероятност
        max_log_prob = np.max(log_prob)
        
        # AIC (Akaike Information Criterion)
        aic = 2 * n_params - 2 * max_log_prob
        
        # BIC (Bayesian Information Criterion)
        bic = np.log(n_data) * n_params - 2 * max_log_prob
        
        # DIC (Deviance Information Criterion)
        mean_log_prob = np.mean(log_prob)
        pD = 2 * (mean_log_prob - max_log_prob)  # Effective number of parameters
        dic = -2 * mean_log_prob + 2 * pD
        
        # WAIC (Watanabe-Akaike Information Criterion) - опростена версия
        waic = -2 * mean_log_prob + 2 * np.var(log_prob)
        
        return {
            'AIC': aic,
            'BIC': bic,
            'DIC': dic,
            'WAIC': waic,
            'max_log_prob': max_log_prob,
            'mean_log_prob': mean_log_prob
        }
    
    def _compare_models(self, model_results: Dict) -> Dict[str, Any]:
        """
        Сравнява модели
        
        Args:
            model_results: Резултати от модели
            
        Returns:
            Сравнение на модели
        """
        model_names = list(model_results.keys())
        
        # Извличаме информационни критерии
        criteria = ['AIC', 'BIC', 'DIC', 'WAIC']
        comparison = {}
        
        for criterion in criteria:
            values = {name: model_results[name]['information_criteria'][criterion] 
                     for name in model_names}
            
            # Намираме най-добрия модел (най-ниска стойност)
            best_model = min(values, key=values.get)
            
            # Пресмятаме относителните разлики
            best_value = values[best_model]
            differences = {name: values[name] - best_value for name in model_names}
            
            comparison[criterion] = {
                'values': values,
                'best_model': best_model,
                'differences': differences
            }
        
        return comparison
    
    def plot_comparison(self, save_path: str = None):
        """
        Графично сравнение на модели
        
        Args:
            save_path: Път за записване
        """
        if not self.results:
            print("Няма резултати за сравнение")
            return
        
        comparison = self.results['comparison']
        model_names = list(self.results['model_results'].keys())
        
        criteria = ['AIC', 'BIC', 'DIC', 'WAIC']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, criterion in enumerate(criteria):
            ax = axes[i]
            
            values = [comparison[criterion]['values'][name] for name in model_names]
            
            bars = ax.bar(model_names, values)
            
            # Маркираме най-добрия модел
            best_model = comparison[criterion]['best_model']
            best_idx = model_names.index(best_model)
            bars[best_idx].set_color('green')
            
            ax.set_xlabel('Модел')
            ax.set_ylabel(criterion)
            ax.set_title(f'Сравнение по {criterion}')
            ax.tick_params(axis='x', rotation=45)
            
            # Добавяме текст за най-добрия модел
            ax.text(0.02, 0.98, f'Най-добър: {best_model}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_summary(self) -> str:
        """
        Генерира текстово резюме на сравнението
        
        Returns:
            Форматирано резюме
        """
        if not self.results:
            return "Няма резултати за показване"
        
        summary = []
        summary.append("=" * 60)
        summary.append("БАЙЕСОВО СРАВНЕНИЕ НА МОДЕЛИ")
        summary.append("=" * 60)
        summary.append("")
        
        comparison = self.results['comparison']
        model_names = list(self.results['model_results'].keys())
        
        for criterion in ['AIC', 'BIC', 'DIC', 'WAIC']:
            summary.append(f"{criterion} СРАВНЕНИЕ:")
            summary.append("-" * 30)
            
            values = comparison[criterion]['values']
            differences = comparison[criterion]['differences']
            best_model = comparison[criterion]['best_model']
            
            for name in model_names:
                marker = " ★" if name == best_model else ""
                summary.append(f"  {name}: {values[name]:.2f} (Δ={differences[name]:.2f}){marker}")
            
            summary.append("")
        
        # Обобщена препоръка
        summary.append("ОБОБЩЕНА ПРЕПОРЪКА:")
        summary.append("-" * 30)
        
        best_counts = {}
        for criterion in ['AIC', 'BIC', 'DIC', 'WAIC']:
            best_model = comparison[criterion]['best_model']
            best_counts[best_model] = best_counts.get(best_model, 0) + 1
        
        overall_best = max(best_counts, key=best_counts.get)
        summary.append(f"Най-добър общ модел: {overall_best}")
        summary.append(f"Брой критерии в полза: {best_counts[overall_best]}/4")
        
        return "\n".join(summary)


def test_mcmc_bayesian():
    """
    Тестова функция за MCMC и Байесов анализ
    """
    # Генерираме тестови данни
    np.random.seed(42)
    true_a, true_b = 2.0, 0.5
    x = np.linspace(0, 10, 20)
    y_true = true_a * x + true_b
    y_obs = y_true + np.random.normal(0, 0.1, len(x))
    
    # Дефинираме модел
    def log_likelihood(params, data):
        a, b, sigma = params
        x_data, y_data = data
        model = a * x_data + b
        return -0.5 * np.sum((y_data - model)**2 / sigma**2 + np.log(2 * np.pi * sigma**2))
    
    def log_prior(params):
        a, b, sigma = params
        if -10 < a < 10 and -10 < b < 10 and 0 < sigma < 1:
            return 0.0
        return -np.inf
    
    # Стартираме MCMC
    analyzer = MCMCBayesianAnalyzer(n_walkers=20, n_steps=500, n_burn=100)
    
    def log_probability(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, (x, y_obs))
    
    result = analyzer.run_mcmc(
        log_probability,
        np.array([1.0, 0.0, 0.1]),
        [(-10, 10), (-10, 10), (0.01, 1.0)]
    )
    
    print("MCMC резултати:")
    print(f"Приемане: {result['acceptance_rate']:.2f}")
    print(f"Най-добри параметри: {analyzer.get_best_fit_params()['median_params']}")
    print(f"Истински параметри: [{true_a}, {true_b}, 0.1]")
    
    # Показваме графики
    analyzer.plot_chains(['a', 'b', 'σ'])
    analyzer.plot_corner(['a', 'b', 'σ'])


if __name__ == "__main__":
    test_mcmc_bayesian() 