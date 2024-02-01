# Import
## Batteries
import os
import sys
import tempfile
## 3rd party
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
import keras
import streamlit as st
## App
from rnatargeting.utils import encoded_nuc_to_str, read_byte_string
from rnatargeting.linearfold import run_linearfold, make_guide_library_features, linearfold_integrate_results
from rnatargeting.models.models import guide_nolin_ninef_model, guide_nolin_threef_model
from rnatargeting.dataset.generators import CNN_sequence_input_dataset, CNN_all_features_input_dataset

# Set random seeds
tf.random.set_seed(0)
np.random.seed(0)

# Functions
@st.cache_data
def run_pred(fpath, outfile=None):
    """
    Main function for running prediction on a FASTA file.
    Args:
        fpath: path to FASTA file
        outfile: path to output CSV file
    Returns:
        pred_df: dataframe of predictions
    """
    # Create a temporary directory
    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_name = tmpdir.name

    # If input is a bytes string, write to file
    if isinstance(fpath, bytes):
        fpath = read_byte_string(fpath, tmpdir_name)  

    # Make guide library and Linearfold input:
    feature_files = make_guide_library_features(fpath, tmpdir_name)

    # Run LinearFold
    fpath_prefix = os.path.join(
        tmpdir_name, os.path.basename(os.path.splitext(fpath)[0])
    )
    ## guide mfe input
    guide_l_out = fpath_prefix + "_linfold_guides_output.txt"
    run_linearfold(feature_files['guide_input'], guide_l_out)

    ## target with 15nt flanks
    target_fl_out = fpath_prefix + "_linfold_target_flank15_output.txt"
    run_linearfold(feature_files['target_flank_input'], target_fl_out)

    ## target with constraints
    target_fl_c_out = fpath_prefix + "_linfold_constraints_target_flank15_output.txt"
    run_linearfold(feature_files['target_flank_c_input'], target_fl_c_out, params = ['--constraints'])

    # Integrate Linearfold results
    feature_f = linearfold_integrate_results(
        feature_files['guide_library'], 
        guide_l_out,
        target_fl_out,
        target_fl_c_out,
        fpath_prefix
    )

    # Predict guide efficiency using the CNN model
    result_f = predict_ensemble_test(
        dataset_name='CNN_sequence_input', 
        model_name='guide_nolin_threef',
        saved='saved_model/sequence_only_input_3f',
        testset_path=feature_f,
        guidelength=30,
        flanklength=15
    )

    # Parse prediction results
    pred_df = parse_prediction_results(feature_files['guide_library'], result_f)

    # Clean up
    sys.stderr.write(f"Cleaning up temporary files...\n")
    tmpdir.cleanup()

    # Write output
    if outfile is not None:
        outdir = os.path.dirname(fpath)
        if outdir != '' or outdir != '.':
            os.makedirs(outdir, exist_ok=True)
        pred_df.to_csv(outfile, index=False)
        sys.stderr.write(f"  Predictions written: {outfile}\n")

    # Filter to top N for each `transcript id` group
    #pred_df = pred_df.groupby('transcript id').head(top_n)

    # Return dataframe
    return pred_df

def logits_mean_absolute_error(y_true, y_pred):
    """
    Custom loss function for regression: mean absolute error of logits
    """
    y_pred = tf.sigmoid(y_pred)
    return keras.metrics.mean_absolute_error(y_true, y_pred)

def logits_mean_squared_error(y_true, y_pred):
    """
    Custom loss function for regression: mean squared error of logits
    """
    y_pred = tf.sigmoid(y_pred)
    return keras.metrics.mean_squared_error(y_true, y_pred)

def create_model(model_name, guidelength):
    """
    Create a Keras model, based on the model name.
    Args:
        model_name (str): name of the model
        guidelength (int): length of the guide
    Returns:
        model (keras.Model): Keras model
    """
    if model_name == 'guide_nolin_threef':
        model = guide_nolin_threef_model(guidelength)
    elif model_name == 'guide_nolin_ninef':
        model = guide_nolin_ninef_model(guidelength)
    else:
        raise ValueError(f'Model name not recognized: {model_name}')
    return model

def create_dataset_generator(dataset_name, testset_path):
    """
    Create a dataset generator, based on the dataset name.
    The dataset generator is used to load the test set to the model.
    Args:
        dataset_name (str): name of the dataset generator
        testset_path (str): path to test set
    Returns:
        dataset (BaseDatasetGenerator): dataset generator
    """
    if dataset_name == 'CNN_sequence_input':
        dataset = CNN_sequence_input_dataset(testset_path)
    elif dataset_name == 'CNN_all_features_input':
        dataset = CNN_all_features_input_dataset(testset_path)
    else:
        raise ValueError(f'Dataset name not recognized: {dataset_name}')
    return dataset

def parse_prediction_results(feature_df_file, pred_df_file):
    """
    Parse model prediction results and create a dataframe
    of predictions sorted by rank.
    Args:
        feature_df_file (str): path to feature dataframe
        pred_df_file (str): path to prediction dataframe
    Returns:
        df_sorted (pd.DataFrame): sorted dataframe
    """
    sys.stderr.write('Parsing prediction results...\n')
    # Load guide feature dataframe
    df_info= pd.read_csv(feature_df_file)[['transcript id','guide','target_pos_list','target pos num']]
    # Load prediction dataframe)
    df_pred = pd.read_csv(pred_df_file)[['spacer sequence','predicted_value_sigmoid']]

    # Merge results
    df_com = (df_info.merge(df_pred, left_on='guide', right_on='spacer sequence')).drop(columns=['spacer sequence'])
    df_com['rank'] = df_com.groupby("transcript id")["predicted_value_sigmoid"].rank(ascending=False)
    df_sorted = df_com.sort_values(by=['transcript id','rank'])

    # return data.frame
    return df_sorted

def predict_ensemble_test(dataset_name, model_name, saved, testset_path, 
                          guidelength, flanklength, regression=False):
    """
    Conduct model prediction on a test set using a new or saved model.
    An ensemble of models is used for prediction.
    The average of the predictions is taken as the final prediction.
    Args:
        dataset_name (str): name of the dataset generator
        model_name (str): name of the model
        saved (str): path to saved model
        testset_path (str): path to test set
        guidelength (int): length of the guide
        flanklength (int): length of the flanking sequence
        regression (bool): regression or classification
    Returns:
        outfile (str): path to output CSV file of predictions
    """
    # Status
    sys.stderr.write(f"Running ensemble prediction...\n")

    # Create dataset generator for loading test set
    test_dataset = create_dataset_generator(dataset_name, testset_path)
    # Run prediction
    if saved is None:
        # Create new model
        sys.stderr.write(f"No saved model provided; creating new model...\n")
        model = create_model(model_name, guidelength=guidelength)
        outfile = None
    else:
        # Load saved model
        sys.stderr.write(f"Loading saved model...\n")
        # Get number of folds
        if model_name == 'guide_nolin_ninef':
            k_folds = 9
        elif model_name == 'guide_nolin_threef':
            k_folds = 3
        else:
            raise ValueError(f'Model name not recognized: {model_name}')
        # Load models and predict for each fold
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
        ## Calc mean across ensemble members
        predict_mean = np.mean(predict_allf, axis=0)
        ## Calc sigmoid of mean
        outputs = np.array(list(tf.sigmoid(predict_mean).numpy().flat))
        # Parse the test set
        test_inputs = [inputs for (inputs, label) in test_dataset.unbatch()]

        # Parse inputs
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
        prefix = os.path.splitext(testset_path)[0]
        outfile = f'{prefix}_guide_prediction_ensemble.csv'
        test_df.to_csv(outfile)

    # Status
    sys.stderr.write(f"  Predictions written to {outfile}\n")
    return outfile

