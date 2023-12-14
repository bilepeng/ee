# Approaching Globally Optimal Energy Efficiency in Interference Networks via Machine Learning

![GitHub](https://img.shields.io/github/license/bilepeng/ee)
[![DOI](https://img.shields.io/badge/doi-10.1109/TWC.2023.3269770-informational)](https://doi.org/10.1109/TWC.2023.3269770)
[![arXiv](https://img.shields.io/badge/arXiv-2212.12329-informational)](https://arxiv.org/abs/2212.12329)

This repository is accompanying the paper "Approaching Globally Optimal Energy
Efficiency in Interference Networks via Machine Learning" (Bile Peng,
Karl-Ludwig Besser, Ramprasad Raghunath and Eduard A. Jorswieck, IEEE
Transactions on Wireless Communications, vol. 22, no. 12, pp. 9313-9326, Dec.
2023. [doi:10.1109/TWC.2023.3269770](https://doi.org/10.1109/TWC.2023.3269770),
[arXiv:2212.12329](https://arxiv.org/abs/2212.12329)).


## File List

The following files are provided in this repository:

- `main.py`: Main file to train the model.
- `test.py`: Script to test the saved model on the validation data set.
- `core.py`: Core setup and functionality.
- `rastrigin.py`: Python module containing the benchmark functions.
- `data/`: Directory containing the training and validation data sets.
- `results/`: Directory containing the results model from the paper.



## Usage

Make sure that you have [Python3](https://www.python.org/downloads/) and all
necessary libraries installed on your machine.

Run `python main.py` with the following arguments to train the model:

- `record`: `True` if you want to save the tensorboard log and trained models in a folder named after date and time of the beginning of training, `False` otherwise.
- `pmax`: maximum transmit power.

Run `python test.py` to test the saved model on the validation data set.


## Acknowledgements

This research was supported by the Federal Ministry of Education and Research
Germany (BMBF) as part of the 6G Research and Innovation Cluster (6G-RIC) under
Grant 16KISK031.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```bibtex
@article{Peng2023,
  author = {Peng, Bile and Besser, Karl-Ludwig and Raghunath, Ramprasad and Jorswieck, Eduard A.},
  title = {Approaching Globally Optimal Energy Efficiency in Interference Networks via Machine Learning},
  journal = {IEEE Transactions on Wireless Communications},
  year = {2023},
  month = {12},
  volume = {22},
  number = {12},
  pages = {9313--9326},
  publisher = {IEEE},
  archiveprefix = {arXiv},
  eprint = {2212.12329},
  primaryclass = {eess.SP},
  doi = {10.1109/TWC.2023.3269770},
}
```
