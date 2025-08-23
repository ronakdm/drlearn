# drlearn

`drlearn`is a library for incorporating distributionally robust optimization seamlessly into estimators that use the `scikit-learn` interface. The focus is on spectral risk measure-based learning which is described in detail in this [AISTATS 2023 paper](https://proceedings.mlr.press/v206/mehta23b.html) and this [ICLR 2024 paper](https://openreview.net/forum?id=TTrzgEZt9s) (Spotlight Presentation). Distributionally robust objectives apply a sample reweighting to the observed training data within each mini-batch in order to robustify models against distribution shifts that occur at test time. This package parallels a similar one, called [Deshift](https://ronakdm.github.io/deshift/), which is built for machine learning workflows based on `torch` (as opposed to `scikit-learn`).

## Installation

You can install `drlearn` by running the following on the command line from the
root folder:
```
pip install -e .
```
To build the docs, additional dependencies can be run using `pip install -e .[docs]`.

## Quickstart

First, we construct a distributionally robust objective that inputs a vector of losses and returns a weighted average of its entries that upweighs its larger values. That is, for $l \in \mathbb{R}^n$, the mapping

$$
    l \mapsto \operatorname{max}_{q \in \mathcal{Q}(\sigma)} \langle q, l \rangle - \nu \operatorname{Pen}(q),
$$

where $\mathcal{Q}(\sigma)$ is the convex hull of all permutations of probability weights $\sigma = (\sigma_1, \ldots, \sigma_n)$ (called the *spectrum*), $\nu \geq 0$ is the *shift cost*, and $\operatorname{Pen}(\cdot)$ is the *penalty*. All of these have default values, and if the user wishes to change them, the primary decision to make is choosing the spectrum $\sigma$. This is parameterized by a univariate quantity, which may be selected as a hyperparameter:
```
from drlearn import make_extremile_spectrum, Ridge

n = 100
X = np.random.normal(size=(n, 10))
y = np.random.normal(size=(n,))


spectrum = make_extremile_spectrum(n, 2.0)
weight_decay = 0.01 # l2 regularization parameter
model = Ridge(spectrum=spectrum, weight_decay=weight_decay).fit(X, y)
```
The variable `model` is now an estimator that has a `predict` method. The `BinaryLogisticRegression` and `MultinomialLogisticRegression` estimators also have `predict_proba` methods. Please refer to `docs/quickstart.ipynb` for a more detailed example.

## Documentation

The documentation is available [here](https://ronakdm.github.io/drlearn/).

## Contributing

If you find any bugs, please raise an issue on GitHub.
If you would like to contribute, please submit a pull request.
We encourage and highly value community contributions.

## Citation

If you find this package useful, or you use it in your research, please cite:

```
@inproceedings{mehta2023stochastic,
    title={{Stochastic Optimization for Spectral Risk Measures}},
    author={Mehta, Ronak and Roulet, Vincent and Pillutla, Krishna and Liu, Lang and Harchaoui, Zaid},
    booktitle={AISTATS},
    year={2023},
}
@inproceedings{mehta2024distributionally,
    title={{Distributionally Robust Optimization with Bias and Variance Reduction}},
    author={Mehta, Ronak and Roulet, Vincent and Pillutla, Krishna and Harchaoui, Zaid},
    booktitle={ICLR},
    year={2024},
}
```


## Acknowledgments

This work was supported by NSF DMS-2023166, CCF-2019844, DMS-2134012, NIH, and the Office of the Director of National Intelligence (ODNI)’s IARPA program via 2022-22072200003. Part of this work was done while Zaid Harchaoui was visiting the Simons Institute for the Theory of Computing. The views and conclusions contained herein are those of the authors and should not be interpreted as representing the official views of ODNI, IARPA, or the U.S. Government.




