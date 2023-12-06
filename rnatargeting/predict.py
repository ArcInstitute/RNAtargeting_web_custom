# Import
import os
import sys
#import importlib
import pkg_resources
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
import keras
#from dataset import find_dataset_generator_using_name
#from options.options import get_arguments
#from rnatargeting.utils import *
from rnatargeting.utils import encoded_nuc_to_str
from rnatargeting.models.find import find_model_using_name
from rnatargeting.models.models import guide_nolin_ninef_model, guide_nolin_threef_model
from rnatargeting.dataset.generators import CNN_sequence_input_dataset, CNN_all_features_input_dataset

# Set random seeds
tf.random.set_seed(0)
np.random.seed(0)


# Functions
def find_dataset_generator_using_name(dataset_generator_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    dataset_filename = "dataset." + dataset_generator_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset_generator = None
    target_dataset_name = dataset_generator_name + '_dataset'
    for name, potential_model_creator in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset_generator = potential_model_creator

    if dataset_generator is None:
        print("In %s.py, there should be a func with name that matches %s in lowercase." % (
            dataset_filename, target_dataset_name))
        exit(0)

    return dataset_generator

def logits_mean_absolute_error(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    return keras.metrics.mean_absolute_error(y_true, y_pred)

def logits_mean_squared_error(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    return keras.metrics.mean_squared_error(y_true, y_pred)

def wbce(y_true, y_pred, weight1 = 1, weight0 = 1) :
    y_true = tf.keras.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = tf.keras.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
    return tf.keras.mean( logloss, axis=-1)

def create_model(model_name, guidelength):
    if model_name == 'guide_nolin_threef':
        model = guide_nolin_threef_model(guidelength)
    elif model_name == 'guide_nolin_ninef':
        model = guide_nolin_ninef_model(guidelength)
    else:
        raise ValueError(f'Model name not recognized: {model_name}')
    return model

def create_dataset_generator(dataset_name, testset_path):
    if dataset_name == 'CNN_sequence_input':
        dataset = CNN_sequence_input_dataset(testset_path)
    elif dataset_name == 'CNN_all_features_input':
        dataset = CNN_all_features_input_dataset(testset_path)
    else:
        raise ValueError(f'Dataset name not recognized: {dataset_name}')
    return dataset

def predict_ensemble_test(dataset_name, model_name, saved, testset_path, guidelength, flanklength, regression=False):
    sys.stderr.write(f"Running ensemble prediction...\n")

    # Check for saved model
    test_dataset = create_dataset_generator(dataset_name, testset_path)
    if saved is None:
        sys.stderr.write(f"No saved model provided; creating new model...\n")
        #return "No saved model provided"
        # Create model
        #model_generator = find_model_using_name(model)
        model = create_model(model_name, guidelength=guidelength)
        #dataset_generator = find_dataset_generator_using_name(dataset)
        outfile = None
    else:
        sys.stderr.write(f"Loading saved model...\n")
        # get folds
        if model_name == 'guide_nolin_ninef':
            k_folds = 9
        elif model_name == 'guide_nolin_threef':
            k_folds = 3
        else:
            raise ValueError(f'Model name not recognized: {model_name}')
        # load models and predict for each fold
        predict_allf = []
        for k in range(k_folds):
            model_path = os.path.join(saved, f'fold_{k}')
            # Regression or classification?
            if regression == True:
                model = keras.models.load_model(
                    model_path,custom_objects={
                        'logits_mean_absolute_error': logits_mean_absolute_error,
                        'logits_mean_squared_error': logits_mean_squared_error
                    }
                )
            else:
                model = keras.models.load_model(model_path)
            # Predict
            output = np.array(model.predict(test_dataset).flat)
            predict_allf.append(output)

        # Format
        predict_allf = np.array(predict_allf)
        ## mean across ensemble members
        predict_mean = np.mean(predict_allf, axis=0)
        ## sigmoid
        outputs = np.array(list(tf.sigmoid(predict_mean).numpy().flat))
        # parse test_dataset
        test_inputs = [inputs for (inputs, label) in test_dataset.unbatch()]


        # Parsing inputs
        if len(test_inputs[0]) == 2:
            test_sequences = [np.array(sequences) for (sequences, features) in test_inputs]
            test_features = [features for (sequences, features) in test_inputs]
        else:
            test_sequences = [np.array(sequences[0]) for sequences in test_inputs]

        # Convert one-hot encoded sequences to strings
        test_predic =[]
        for i in range(len(outputs)):
            nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
            test_predic.append([nuc_sequence, outputs[i]])
        test_df = pd.DataFrame(test_predic, columns = ['spacer sequence', 'predicted_value_sigmoid'])
        
        # Save output
        dataset_folder = os.path.join('results', dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        prefix = os.path.splitext(os.path.basename(testset_path))[0]
        outfile = os.path.join(dataset_folder, f'{prefix}_guide_prediction_ensemble.csv')
        test_df.to_csv(outfile)

    sys.stderr.write(f"  Predictions written to {outfile}\n")
    return outfile



