[![DOI:10.1016/j.cels.2023.11.006](http://img.shields.io/badge/DOI-10.1016/j.cels.2023.11.006-brightgreen.svg)](https://doi.org/10.1016/j.cels.2023.11.006)
[![website](https://img.shields.io/badge/website-live-brightgreen)](https://arcinstitute.org/tools/cas13d)

RNAtargeting_web_custom
=======================

Source code for the "custom" feature of the [cas13d RNA-targeting web app](https://arcinstitute.org/tools/cas13d).

> WARNING: this is a work in progress; the API could change at any time

# Usage 

## Install

```bash
pip install -r requirements.txt
```

## [optional] Run tests

```bash
pytest -s tests
```

## Run the app

```bash
streamlit run app.py
```

# Citation

> Wei, J., Lotfy, P., Faizi, K., Baungaard, S., Gibson, E., Wang, E., Slabodkin, H., Kinnaman, E., Chandrasekaran, S., Kitano, H., Durrant, M. G., Duffy, C. V., Pawluk, A., Hsu, P. D., & Konermann, S. (2023).
**Deep learning and CRISPR-Cas13d ortholog discovery for optimized RNA targeting**. _Cell systems_, 14(12), 1087–1102.e13. https://doi.org/10.1016/j.cels.2023.11.006
