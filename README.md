# Code Source for the Paper "Multivariate Functional Linear Discriminant Analysis (MUDRA) for the Classification of Short Time Series with Missing Data"

## Installation

We advise to use a virtual environment, either with Conda or VirtualEnv. Then run the following command:

```bash
git clone https://github.com/SirUnknown2/MUDRA.git
cd MUDRA/
python -m pip install --upgrade setuptools
python -m pip install -r pip/requirements.txt
python setup.py install --user
```

## Example

The class *MUDRA* is defined like a scikit-learn module, that is

- To import the *MUDRA* class:

```python
from MUDRA import MUDRA
```

- To fit the model on training data (X,y) (for r=8, b=9 and 300 iterations for the last optimization step):

```python
model = MUDRA(r=8, n_iter=300, nBasis=9).fit(X, y)
```

- To perform dimension reduction on new data (X):

```python
x = model.transform(X)
```

- To predict labels on new data (X):

```python
y = model.predict(X)
```

## Reproduce the results shown in the paper

Please check out the interactive Jupyter notebook "*FLDA_ECM-Words.ipynb*". After installing Jupyter Notebook, please run the following command:

```bash
jupyter notebook FLDA_ECM-Words.ipynb 
```

## Citations

If you use *MUDRA* in academic research, please cite it as follows

```
@unpublished{Bordoloi2024, 
    year = {2024}, 
    note = {Under review}, 
    author = {Rahul Bordoloi, Clémence Réda, Orell Trautmann, Saptarshi Bej and Olaf Wolkenhauer}, 
    title = {Multivariate Functional Linear Discriminant Analysis (MUDRA) for the Classification of Short Time Series with Missing Data}, 
} 

```

The citation for the ``Articulary Word Recognition'' data set (available in folder "*datasets/*") is

```
@article{ruiz_great_2021,
	title = {The great multivariate time series classification bake off: a review and experimental evaluation of recent algorithmic advances},
	volume = {35},
	issn = {1573-756X},
	doi = {10.1007/s10618-020-00727-3},
	number = {2},
	journal = {Data Min Knowl Disc},
	author = {Ruiz, Alejandro Pasos and Flynn, Michael and Large, James and Middlehurst, Matthew and Bagnall, Anthony},
	month = mar,
	year = {2021},
	pages = {401--449},
}

```

Original link to the freely available dataset is [here](http://www.timeseriesclassification.com/description.php?Dataset=ArticularyWordRecognition).
