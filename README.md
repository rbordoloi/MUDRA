# Source Code for the Paper "Multivariate Functional Linear Discriminant Analysis (MUDRA) for the Classification of Short Time Series with Missing Data"
 
 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multivariate-functional-linear-discriminant/time-series-classification-on-2)](https://paperswithcode.com/sota/time-series-classification-on-2?p=multivariate-functional-linear-discriminant)

## Installation

We advise to use a virtual environment, either with Conda or VirtualEnv. Then run the following command:

```bash
git clone https://github.com/rbordoloi/MUDRA.git
cd MUDRA/
# If using pip
python -m pip install --upgrade setuptools
python -m pip install -r pip/requirements.txt
# If using conda
cond env create --name mudra --file=environment.yml
python setup.py install --user
```

The sample dataset is included in the `datasets` directory. To regenerate the dataset from the original source, run `datasetGeneration.py`.

## Example

The class *MUDRA* is defined like a scikit-learn module, that is

- To import the *MUDRA* class:

```python
from MUDRA import MUDRA
```

The model accepts input `X` as a pandas DataFrame of shape `(n_samples, n_features)` and `y` as list of class labels. Each cell of the DataFrame has a pandas Series object corresponding to the time series for one feature of one sample. Each Series object is indexed by the time points for which observations were recorded. Missing features are denoted by `np.nan` objects.
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

- To predict scores on new data (X):

```python
y = model.predict_proba(X)
```

## Reproduce the results shown in the paper

Please check out the interactive Jupyter notebooks "*synthetic.ipynb*" and "*real.ipynb*". After installing Jupyter Notebook, please run the following commands:

```bash
jupyter notebook real.ipynb
jupyter notebook synthetic.ipynb
```

## Citations

If you use *MUDRA* in academic research, please cite it as follows

```
@misc{bordoloi2024multivariate,
      title={Multivariate Functional Linear Discriminant Analysis for the Classification of Short Time Series with Missing Data},
      author={Rahul Bordoloi and Clémence Réda and Orell Trautmann and Saptarshi Bej and Olaf Wolkenhauer},
      year={2024},
      eprint={2402.13103},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
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
