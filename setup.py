from setuptools import setup, find_packages

NAME = "MUDRA"
VERSION = '1.0.0'

setup(
    name=NAME,
    version=VERSION,
    author='Rahul Bordoloi',
    author_email='rahul.bordoloi@uni-rostock.com',
    url='https://github.com/rbordoloi/MUDRA',
    license_file = ('LICENSE'),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Programming Language :: Python :: 3"
    ],
    keywords='',
    description='Multivariate Functional Linear Discriminant Analysis (MUDRA) for the Classification of Short Time Series with Missing Data.',
    long_desciption=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='MUDRA'),
    package_dir={'': 'MUDRA'},
    python_require='>=3.8.5',
    install_requires=[
        'numpy>=1.24.0',
        'scikit-base>=0.6.0',
        'scikit-fda>=0.9',
        'scipy>=1.11.2',
        'tensorly>=0.8.1',
        'sktime>=0.26.0',
        'tqdm==4.40.0',
        'scikit-learn>=1.4.1',
        'pandas>=2.1.4'
    ],
    entry_points={}
)
