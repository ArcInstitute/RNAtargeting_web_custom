# RNAtargeting_web_custom
source code of the custom page for the RNA targeting website - Cas13d guide design

## Downloading and Setting up

Clone the repository

```bash
git clone https://github.com/jingyi7777/RNAtargeting_web_custom.git
```

Compile the Linearfold binary

```bash
cd RNAtargeting_web_custom
git clone https://github.com/LinearFold/LinearFold.git
cd LinearFold
make
```

Install python dependencies

```bash
pip3 install -r requirements.txt
```

## Test locally

```bash
python app.py
#flask --app asana_webhooks run --port ${PORT} --debug
```

## Build the Docker image

```bash
IMG_NAME=rna-targeting-web-custom
IMG_VERSION=0.1.0
docker build --platform linux/amd64 -t ${IMG_NAME}:${IMG_VERSION} .
```

## Contents
* `app.py`: FLASK app script
* `templates/`: HTML templates -- "header.html" contains the header, scripts and CSS styles; "custom.html" is the custom input page; "custom_results.html" is the custom result output page.
* `predict.py`: Function for custom CasRx guide design, which involves the use of the scripts in `scripts/` and `predict_ensemble_test.py`
* `scripts/`: Scripts for CasRx guide design for custom input sequences
* `dataset/`: Model input dataset and features for custom CasRx guide design
* `models/`;`saved_model/`: CNN models and saved models for custom CasRx guide design
* `utils.py`: Provides useful utilities for the CNN model
* `static/`: Contains images for the website and the prediction results for guide design


# TODO

* Add class/function documentation
* Remove old, commented-out code
* Re-organize code for clarity
* Add exception handling
* Add test datasets
* Add unit/application tests
* Add package dependency versions
* Change how LinearFold is installed
* Add a ci/cd pipeline
* Convert Flask app to FastAPI?
  * Call API from the tool-hub website
* Create a python package?
* Create a well-developed CLI?

## Stats

* No. of Python files: 16
* No. of functions: 97
* Lines of code: 3212 