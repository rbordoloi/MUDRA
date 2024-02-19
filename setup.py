from setuptools import setup, find_packages

NAME = "MUDRA"
VERSION = '1.0.0'

setup(
    name=NAME,
    version=VERSION,
    autho='Rahul Bordoloi',
    author_email='rahul.bordoloi@uni-rostock.com',
    url='https://github.com/SirUnknown2/MUDRA',
    license_file = ('LICENSE'),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],
    keywords='',
    description='Multivariate fUnctional linear DiscRiminant Analysis',
    long_desciption=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='MUDRA'),
    package_dir={'': 'MUDRA'},
    python_require='>3.8.5',
    install_requires=[
        'numpy>=1.24.3',
        'scikit-base>=0.6.0',
        'scikit-fda>=0.9',
        'scipy>=1.11.2',
        'tensorly>=0.8.1'
    ],
    entry_points={}
)
