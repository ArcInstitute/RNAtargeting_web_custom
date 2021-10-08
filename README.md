# RNAtargeting_web_custom
source code of the custom page for the RNA targeting website - Cas13d guide design

## Downloading and Setting up
* Clone the repository
```
git clone https://github.com/jingyi7777/RNAtargeting_web_custom.git
```
* Setting up Linearfold: (Dependencies: GCC 4.8.5 or above; Python 2.7)
```
cd RNAtargeting_web_custom
git clone https://github.com/LinearFold/LinearFold.git
cd LinearFold
make
```
* Setting up Tensorflow2 and other packages: (Dependencies: Python 3)
```
pip3 install -r requirements.txt
```

## Testing locally
```
python app.py
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

