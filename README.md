# drlearn

`drlearn`is a library for incorporating distributionally robust optimization seamlessly into estimators that use the `scikit-learn` interface.

## Installation

You can install `drlearn` by running the following on the command line from the
root folder:
```
pip install -e .
```
Additional dependencies to run the example in `examples/*.ipynb` can be installed using `pip install -e .[examples]`. To build the docs, additional dependencies can be run using `pip install -e .[docs]`.

## Quickstart



## Documentation

The documentation is available [here](https://ronakdm.github.io/deshift/).

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




