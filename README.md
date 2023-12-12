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

> Deep learning and CRISPR-Cas13d ortholog discovery for optimized RNA targeting.
Jingyi Wei, Peter Lotfy, Kian Faizi, Sara Baungaard, Emily Gibson, Eleanor Wang, Hannah Slabodkin, Emily Kinnaman, Sita Chandrasekaran, Hugo Kitano, Matthew G. Durrant, Connor V. Duffy, Patrick D. Hsu, Silvana Konermann
bioRxiv 2021.09.14.460134; doi: https://doi.org/10.1101/2021.09.14.460134