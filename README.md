RNAtargeting_web_custom
=======================

Source code of the custom page for the RNA targeting website - Cas13d guide design

> WARNING: this is a work in progress; the API could change at any time

# Run the app locally

```bash
streamlit run app.py
```

# Dev 

It's best to use the .devcontainer to develop this project.
It has all the dependencies installed and configured.

## Run tests

```bash
pytest -s tests
```

***

# OLD 

## Contents
* `app.py`: FLASK app script
* `templates/`: HTML templates -- "header.html" contains the header, scripts and CSS styles; "custom.html" is the custom input page; "custom_results.html" is the custom result output page.
* `predict.py`: Function for custom CasRx guide design, which involves the use of the scripts in `scripts/` and `predict_ensemble_test.py`
* `scripts/`: Scripts for CasRx guide design for custom input sequences
* `dataset/`: Model input dataset and features for custom CasRx guide design
* `models/`;`saved_model/`: CNN models and saved models for custom CasRx guide design
* `utils.py`: Provides useful utilities for the CNN model
* `static/`: Contains images for the website and the prediction results for guide design


