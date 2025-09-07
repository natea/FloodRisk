"""
Setup configuration for FloodRisk package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    """Read README.md file."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read version from __init__.py
def get_version():
    """Get version from src/floodrisk/__init__.py"""
    version_file = os.path.join('src', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    return '0.1.0'

# Define requirements
install_requires = [
    'torch>=2.0.0',
    'torchvision>=0.15.0',
    'numpy>=1.24.0',
    'scipy>=1.10.0',
    'gdal>=3.6.0',
    'rasterio>=1.3.0',
    'fiona>=1.9.0',
    'shapely>=2.0.0',
    'geopandas>=0.12.0',
    'xarray>=2023.1.0',
    'netcdf4>=1.6.0',
    'pyproj>=3.4.0',
    'pandas>=1.5.0',
    'scikit-learn>=1.2.0',
    'matplotlib>=3.6.0',
    'seaborn>=0.12.0',
    'plotly>=5.13.0',
    'opencv-python>=4.7.0',
    'pillow>=9.4.0',
    'scikit-image>=0.19.0',
    'tensorboard>=2.11.0',
    'pytorch-lightning>=1.9.0',
    'torchmetrics>=0.11.0',
    'richdem>=2.3.0',
    'whitebox>=2.2.0',
    'pyflwdir>=0.5.0',
    'fastapi>=0.95.0',
    'uvicorn[standard]>=0.20.0',
    'pydantic>=1.10.0',
    'python-multipart>=0.0.6',
    'sqlalchemy>=2.0.0',
    'alembic>=1.10.0',
    'redis>=4.5.0',
    'psycopg2-binary>=2.9.0',
    'python-dotenv>=1.0.0',
    'pydantic-settings>=2.0.0',
    'click>=8.1.0',
    'loguru>=0.6.0',
    'prometheus-client>=0.16.0',
    'dask[complete]>=2023.1.0',
    'joblib>=1.2.0',
]

dev_requires = [
    'pytest>=7.2.0',
    'pytest-asyncio>=0.20.0',
    'pytest-cov>=4.0.0',
    'pytest-mock>=3.10.0',
    'httpx>=0.23.0',
    'black>=23.1.0',
    'isort>=5.12.0',
    'flake8>=6.0.0',
    'mypy>=1.0.0',
    'pre-commit>=3.0.0',
    'sphinx>=6.0.0',
    'sphinx-rtd-theme>=1.2.0',
    'jupyter>=1.0.0',
    'ipykernel>=6.20.0',
    'jupyterlab>=3.6.0',
]

setup(
    name='floodrisk',
    version=get_version(),
    description='Flood depth prediction system using machine learning and hydrological modeling',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    
    # Author and contact information
    author='FloodRisk Development Team',
    author_email='info@floodrisk.com',
    url='https://github.com/yourusername/floodrisk',
    
    # Package configuration
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.11',
    
    # Dependencies
    install_requires=install_requires,
    extras_require={
        'dev': dev_requires,
        'docs': ['sphinx>=6.0.0', 'sphinx-rtd-theme>=1.2.0'],
        'jupyter': ['jupyter>=1.0.0', 'ipykernel>=6.20.0'],
        'cloud': ['boto3>=1.26.0', 'google-cloud-storage>=2.7.0'],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        'floodrisk': [
            'data/*.json',
            'configs/*.yaml',
            'templates/*.html',
        ],
    },
    
    # Entry points
    entry_points={
        'console_scripts': [
            'floodrisk=floodrisk.cli:main',
            'floodrisk-train=floodrisk.training.train:main',
            'floodrisk-predict=floodrisk.prediction.predict:main',
            'floodrisk-api=floodrisk.api.main:main',
        ],
    },
    
    # Classification
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Hydrology',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    
    # Keywords
    keywords='flood prediction machine learning hydrology geospatial pytorch',
    
    # Project URLs
    project_urls={
        'Documentation': 'https://floodrisk.readthedocs.io/',
        'Source': 'https://github.com/yourusername/floodrisk',
        'Tracker': 'https://github.com/yourusername/floodrisk/issues',
    },
    
    # Additional metadata
    zip_safe=False,
    platforms=['any'],
)