"""
Процесори за сурови данни
========================

Този модул обработва сурови данни от:
- SH0ES (Supernovae H0 for the Equation of State)
- Pantheon+ (Supernovae Catalog)
- Други космологични измервания

БЕЗ ΛCDM адаптации - само сурови наблюдения!

Автор: Система за анализ на нелинейно време
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import re
import warnings
from typing import Dict, List, Tuple, Any, Optional
import os

warnings.filterwarnings('ignore')


class RawDataProcessor:
    """
    Основен клас за обработка на сурови данни
    """
    
    def __init__(self, data_path: str = None):
        """
        Инициализация на процесора
        
        Args:
            data_path: Път към директорията с данни
        """
        self.data_path = data_path or r"D:\MyPRJ\Python\NotLinearTime\test_2\data"
        self.loaded_data = {}
        
    def load_pantheon_plus_data(self) -> Dict[str, Any]:
        """
        Зарежда сурови данни от Pantheon+
        
        Returns:
            Речник с данни
        """
        pantheon_path = os.path.join(self.data_path, "Pantheon+_Data")
        
        data = {}
        
        # 1. Зареждаме основните данни от различните директории
        for subdir in ['1_DATA', '2_CALIBRATION', '3_SALT2', '4_DISTANCES_AND_COVAR']:
            subdir_path = os.path.join(pantheon_path, subdir)
            if os.path.exists(subdir_path):
                data[subdir] = self._load_directory_files(subdir_path)
        
        # 2. Основният каталог с резултати
        main_files = self._load_directory_files(pantheon_path)
        data['main'] = main_files
        
        self.loaded_data['pantheon_plus'] = data
        return data
    
    def load_shoes_data(self) -> Dict[str, Any]:
        """
        Зарежда сурови данни от SH0ES
        
        Returns:
            Речник с данни
        """
        shoes_path = os.path.join(self.data_path, "SH0ES_Data")
        
        data = {}
        
        # Зареждаме всички файлове
        for filename in os.listdir(shoes_path):
            file_path = os.path.join(shoes_path, filename)
            
            if filename.endswith('.fits'):
                # FITS файлове
                data[filename] = self._load_fits_file(file_path)
            elif filename.endswith('.dat'):
                # ASCII данни
                data[filename] = self._load_ascii_file(file_path)
            elif filename.endswith('.tex'):
                # LaTeX таблици
                data[filename] = self._load_latex_table(file_path)
            elif filename.endswith('.out'):
                # Изходни файлове
                data[filename] = self._load_ascii_file(file_path)
        
        self.loaded_data['shoes'] = data
        return data
    
    def _load_directory_files(self, directory: str) -> Dict[str, Any]:
        """
        Зарежда всички файлове от директория
        
        Args:
            directory: Път към директорията
            
        Returns:
            Речник с данни
        """
        data = {}
        
        if not os.path.exists(directory):
            return data
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                try:
                    if filename.endswith('.fits'):
                        data[filename] = self._load_fits_file(file_path)
                    elif filename.endswith(('.dat', '.txt', '.csv')):
                        data[filename] = self._load_ascii_file(file_path)
                    elif filename.endswith('.tex'):
                        data[filename] = self._load_latex_table(file_path)
                except Exception as e:
                    print(f"Грешка при зареждане на {filename}: {e}")
        
        return data
    
    def _load_fits_file(self, file_path: str) -> Dict[str, Any]:
        """
        Зарежда FITS файл
        
        Args:
            file_path: Път към файла
            
        Returns:
            Данни от FITS файла
        """
        try:
            with fits.open(file_path) as hdul:
                data = {
                    'header': dict(hdul[0].header),
                    'data': None,
                    'table': None
                }
                
                # Проверяваме дали има данни
                if len(hdul) > 1:
                    if hdul[1].data is not None:
                        data['table'] = Table(hdul[1].data)
                        data['data'] = hdul[1].data
                
                return data
        except Exception as e:
            print(f"Грешка при зареждане на FITS файл {file_path}: {e}")
            return {}
    
    def _load_ascii_file(self, file_path: str) -> Dict[str, Any]:
        """
        Зарежда ASCII файл
        
        Args:
            file_path: Път към файла
            
        Returns:
            Данни от ASCII файла
        """
        try:
            # Опитваме се да определим формата
            with open(file_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(10)]
            
            # Проверяваме дали има header
            has_header = any(line.startswith('#') or 
                           not line.replace('.', '').replace('-', '').replace(' ', '').replace('\t', '').isdigit()
                           for line in first_lines[:3] if line)
            
            # Зареждаме данните
            if has_header:
                data = pd.read_csv(file_path, delim_whitespace=True, comment='#')
            else:
                data = pd.read_csv(file_path, delim_whitespace=True, header=None)
            
            return {
                'dataframe': data,
                'values': data.values,
                'columns': data.columns.tolist()
            }
        except Exception as e:
            print(f"Грешка при зареждане на ASCII файл {file_path}: {e}")
            return {}
    
    def _load_latex_table(self, file_path: str) -> Dict[str, Any]:
        """
        Зарежда LaTeX таблица
        
        Args:
            file_path: Път към файла
            
        Returns:
            Данни от LaTeX таблицата
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Опитваме се да извлечем данните от LaTeX формата
            # Това е опростена версия - може да се подобри
            lines = content.split('\n')
            data_lines = []
            
            for line in lines:
                # Търсим редове с данни (съдържат & и \\)
                if '&' in line and '\\' in line:
                    # Почистваме LaTeX командите
                    clean_line = re.sub(r'\\[a-zA-Z]+', '', line)
                    clean_line = clean_line.replace('&', ' ').replace('\\', '')
                    clean_line = clean_line.strip()
                    
                    if clean_line:
                        data_lines.append(clean_line)
            
            # Опитваме се да парсираме данните
            parsed_data = []
            for line in data_lines:
                try:
                    values = line.split()
                    numeric_values = []
                    for val in values:
                        try:
                            numeric_values.append(float(val))
                        except ValueError:
                            numeric_values.append(val)
                    parsed_data.append(numeric_values)
                except:
                    continue
            
            return {
                'raw_content': content,
                'parsed_data': parsed_data,
                'n_rows': len(parsed_data)
            }
        except Exception as e:
            print(f"Грешка при зареждане на LaTeX файл {file_path}: {e}")
            return {}
    
    def extract_redshift_magnitude_data(self) -> Dict[str, Any]:
        """
        Извлича z и magnitude данни от всички източници
        
        Returns:
            Унифицирани данни за z и magnitude
        """
        unified_data = {
            'pantheon_plus': {},
            'shoes': {},
            'combined': {}
        }
        
        # Pantheon+ данни
        if 'pantheon_plus' in self.loaded_data:
            unified_data['pantheon_plus'] = self._extract_pantheon_z_mag()
        
        # SH0ES данни
        if 'shoes' in self.loaded_data:
            unified_data['shoes'] = self._extract_shoes_z_mag()
        
        # Комбинираме данните
        unified_data['combined'] = self._combine_z_mag_data(unified_data)
        
        return unified_data
    
    def _extract_pantheon_z_mag(self) -> Dict[str, Any]:
        """
        Извлича z и magnitude от Pantheon+ данни
        
        Returns:
            Речник с z и magnitude данни
        """
        data = {}
        
        pantheon_data = self.loaded_data['pantheon_plus']
        
        # Търсим в различните директории
        for subdir, files in pantheon_data.items():
            for filename, file_data in files.items():
                if 'dataframe' in file_data:
                    df = file_data['dataframe']
                    
                    # Търсим колони за z и magnitude
                    z_cols = [col for col in df.columns if 'z' in col.lower() or 'redshift' in col.lower()]
                    mag_cols = [col for col in df.columns if 'mag' in col.lower() or 'mu' in col.lower()]
                    
                    if z_cols and mag_cols:
                        data[f"{subdir}_{filename}"] = {
                            'z': df[z_cols[0]].values,
                            'magnitude': df[mag_cols[0]].values,
                            'z_column': z_cols[0],
                            'mag_column': mag_cols[0],
                            'n_points': len(df)
                        }
        
        return data
    
    def _extract_shoes_z_mag(self) -> Dict[str, Any]:
        """
        Извлича z и magnitude от SH0ES данни
        
        Returns:
            Речник с z и magnitude данни
        """
        data = {}
        
        shoes_data = self.loaded_data['shoes']
        
        for filename, file_data in shoes_data.items():
            if 'dataframe' in file_data:
                df = file_data['dataframe']
                
                # Търсим колони за z и magnitude
                z_cols = [col for col in df.columns if 'z' in str(col).lower() or 'redshift' in str(col).lower()]
                mag_cols = [col for col in df.columns if 'mag' in str(col).lower() or 'mu' in str(col).lower()]
                
                if z_cols and mag_cols:
                    data[filename] = {
                        'z': df[z_cols[0]].values,
                        'magnitude': df[mag_cols[0]].values,
                        'z_column': z_cols[0],
                        'mag_column': mag_cols[0],
                        'n_points': len(df)
                    }
            
            elif 'table' in file_data and file_data['table'] is not None:
                table = file_data['table']
                
                # Търсим колони за z и magnitude
                z_cols = [col for col in table.colnames if 'z' in col.lower() or 'redshift' in col.lower()]
                mag_cols = [col for col in table.colnames if 'mag' in col.lower() or 'mu' in col.lower()]
                
                if z_cols and mag_cols:
                    data[filename] = {
                        'z': table[z_cols[0]].data,
                        'magnitude': table[mag_cols[0]].data,
                        'z_column': z_cols[0],
                        'mag_column': mag_cols[0],
                        'n_points': len(table)
                    }
        
        return data
    
    def _combine_z_mag_data(self, unified_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Комбинира z и magnitude данни от всички източници
        
        Args:
            unified_data: Унифицирани данни
            
        Returns:
            Комбинирани данни
        """
        all_z = []
        all_mag = []
        sources = []
        
        # Pantheon+ данни
        for key, data in unified_data['pantheon_plus'].items():
            if 'z' in data and 'magnitude' in data:
                all_z.extend(data['z'])
                all_mag.extend(data['magnitude'])
                sources.extend(['pantheon_plus'] * len(data['z']))
        
        # SH0ES данни
        for key, data in unified_data['shoes'].items():
            if 'z' in data and 'magnitude' in data:
                all_z.extend(data['z'])
                all_mag.extend(data['magnitude'])
                sources.extend(['shoes'] * len(data['z']))
        
        # Конвертираме в numpy масиви
        all_z = np.array(all_z)
        all_mag = np.array(all_mag)
        sources = np.array(sources)
        
        # Филтрираме валидни данни
        valid_mask = np.isfinite(all_z) & np.isfinite(all_mag) & (all_z > 0)
        
        return {
            'z': all_z[valid_mask],
            'magnitude': all_mag[valid_mask],
            'sources': sources[valid_mask],
            'n_points': np.sum(valid_mask),
            'z_range': (np.min(all_z[valid_mask]), np.max(all_z[valid_mask])),
            'mag_range': (np.min(all_mag[valid_mask]), np.max(all_mag[valid_mask]))
        }
    
    def get_raw_data_summary(self) -> str:
        """
        Генерира резюме на заредените данни
        
        Returns:
            Текстово резюме
        """
        summary = []
        summary.append("=" * 60)
        summary.append("РЕЗЮМЕ НА СУРОВИ ДАННИ")
        summary.append("=" * 60)
        summary.append("")
        
        # Pantheon+ данни
        if 'pantheon_plus' in self.loaded_data:
            summary.append("PANTHEON+ ДАННИ:")
            summary.append("-" * 20)
            
            for subdir, files in self.loaded_data['pantheon_plus'].items():
                summary.append(f"  {subdir}: {len(files)} файла")
                for filename in files.keys():
                    summary.append(f"    - {filename}")
            
            summary.append("")
        
        # SH0ES данни
        if 'shoes' in self.loaded_data:
            summary.append("SH0ES ДАННИ:")
            summary.append("-" * 20)
            
            for filename, data in self.loaded_data['shoes'].items():
                file_type = "FITS" if 'table' in data else "ASCII" if 'dataframe' in data else "LaTeX"
                summary.append(f"  {filename} ({file_type})")
                
                if 'dataframe' in data:
                    df = data['dataframe']
                    summary.append(f"    Размер: {df.shape[0]} x {df.shape[1]}")
                    summary.append(f"    Колони: {df.columns.tolist()}")
                elif 'table' in data and data['table'] is not None:
                    table = data['table']
                    summary.append(f"    Размер: {len(table)} записа")
                    summary.append(f"    Колони: {table.colnames}")
            
            summary.append("")
        
        return "\n".join(summary)
    
    def plot_raw_data_overview(self, save_path: str = None):
        """
        Графичен преглед на сурови данни
        
        Args:
            save_path: Път за записване
        """
        # Извличаме унифицираните данни
        unified_data = self.extract_redshift_magnitude_data()
        
        if not unified_data['combined']:
            print("Няма данни за показване")
            return
        
        combined = unified_data['combined']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Hubble диаграма
        # Създаваме числови кодове за източниците
        unique_sources = np.unique(combined['sources'])
        source_colors = {source: i for i, source in enumerate(unique_sources)}
        color_values = [source_colors[source] for source in combined['sources']]
        
        scatter = axes[0, 0].scatter(combined['z'], combined['magnitude'], 
                                    alpha=0.6, s=20, c=color_values, cmap='viridis')
        axes[0, 0].set_xlabel('Червено отместване z')
        axes[0, 0].set_ylabel('Модулна величина')
        axes[0, 0].set_title('Hubble диаграма (сурови данни)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Добавяме легенда
        from matplotlib.colors import ListedColormap
        import matplotlib.cm as cm
        cmap = cm.get_cmap('viridis')
        for i, source in enumerate(unique_sources):
            axes[0, 0].scatter([], [], c=cmap(i/len(unique_sources)), 
                              label=source, s=20)
        axes[0, 0].legend(loc='upper left', fontsize=8)
        
        # 2. Разпределение на z
        axes[0, 1].hist(combined['z'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Червено отместване z')
        axes[0, 1].set_ylabel('Честота')
        axes[0, 1].set_title('Разпределение на червените отмествания')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Разпределение на magnitude
        axes[1, 0].hist(combined['magnitude'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Модулна величина')
        axes[1, 0].set_ylabel('Честота')
        axes[1, 0].set_title('Разпределение на модулните величини')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Статистики по източници
        unique_sources, counts = np.unique(combined['sources'], return_counts=True)
        axes[1, 1].bar(unique_sources, counts)
        axes[1, 1].set_xlabel('Източник')
        axes[1, 1].set_ylabel('Брой наблюдения')
        axes[1, 1].set_title('Брой наблюдения по източник')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Принтираме статистики
        print("\nСТАТИСТИКИ НА СУРОВИ ДАННИ:")
        print("-" * 30)
        print(f"Общо наблюдения: {combined['n_points']}")
        print(f"Диапазон z: {combined['z_range'][0]:.4f} - {combined['z_range'][1]:.4f}")
        print(f"Диапазон magnitude: {combined['mag_range'][0]:.2f} - {combined['mag_range'][1]:.2f}")
        
        for source, count in zip(unique_sources, counts):
            print(f"{source}: {count} наблюдения")


def test_raw_data_processor():
    """
    Тестова функция за процесора на сурови данни
    """
    # Създаваме процесор
    processor = RawDataProcessor()
    
    # Проверяваме дали данните са достъпни
    data_path = processor.data_path
    pantheon_path = os.path.join(data_path, "Pantheon+_Data")
    shoes_path = os.path.join(data_path, "SH0ES_Data")
    
    if not os.path.exists(data_path):
        print(f"❌ Данните не са достъпни в {data_path}")
        print("🔧 Използваме синтетични данни за тестване...")
        
        # Генерираме синтетични данни
        np.random.seed(42)
        n_points = 1000
        
        # Синтетични redshift данни
        z_synthetic = np.random.exponential(scale=0.5, size=n_points)
        z_synthetic = z_synthetic[z_synthetic < 3.0]  # Ограничаваме до z < 3
        
        # Синтетични magnitude данни (според Standard candles)
        # μ = 5*log10(d_L) + 25 где d_L е luminosity distance
        H0 = 70  # km/s/Mpc
        c = 3e5  # km/s
        
        # Опростен модел за luminosity distance
        d_L = (c / H0) * z_synthetic * (1 + z_synthetic/2)  # Приближение
        magnitude_synthetic = 5 * np.log10(d_L) + 25
        
        # Добавяме шум
        magnitude_synthetic += np.random.normal(0, 0.1, len(magnitude_synthetic))
        
        # Създаваме фалшиви данни
        processor.loaded_data = {
            'synthetic': {
                'test_data': {
                    'z': z_synthetic,
                    'magnitude': magnitude_synthetic,
                    'sources': ['synthetic'] * len(z_synthetic),
                    'n_points': len(z_synthetic),
                    'z_range': (np.min(z_synthetic), np.max(z_synthetic)),
                    'mag_range': (np.min(magnitude_synthetic), np.max(magnitude_synthetic))
                }
            }
        }
        
        print("✅ Синтетични данни генерирани успешно")
        print(f"   Брой точки: {len(z_synthetic)}")
        print(f"   z диапазон: {np.min(z_synthetic):.3f} - {np.max(z_synthetic):.3f}")
        print(f"   magnitude диапазон: {np.min(magnitude_synthetic):.1f} - {np.max(magnitude_synthetic):.1f}")
        
        return {
            'status': 'success_synthetic',
            'processor': processor,
            'data_summary': 'Синтетични данни използвани за тестване',
            'n_points': len(z_synthetic),
            'z_range': (np.min(z_synthetic), np.max(z_synthetic)),
            'mag_range': (np.min(magnitude_synthetic), np.max(magnitude_synthetic))
        }
    
    else:
        # Зареждаме реални данни
        print("Зареждане на Pantheon+ данни...")
        pantheon_data = processor.load_pantheon_plus_data()
        
        print("Зареждане на SH0ES данни...")
        shoes_data = processor.load_shoes_data()
        
        # Показваме резюме
        print("\nРезюме на данните:")
        print(processor.get_raw_data_summary())
        
        # Извличаме z и magnitude данни
        print("\nИзвличане на z и magnitude данни...")
        unified_data = processor.extract_redshift_magnitude_data()
        
        # Показваме графики
        processor.plot_raw_data_overview()
        
        return {
            'status': 'success_real',
            'processor': processor,
            'data_summary': processor.get_raw_data_summary(),
            'unified_data': unified_data
        }


if __name__ == "__main__":
    processor = test_raw_data_processor() 