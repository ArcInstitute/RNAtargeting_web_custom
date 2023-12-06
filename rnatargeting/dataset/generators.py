# import
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from rnatargeting.dataset.utils import *

# functions
def CNN_sequence_input_dataset(testset_path):
    # test set data, read feature file path
    tedf = pd.read_csv(testset_path)
    encoded_guides_te = [one_hot_encode_sequence(guide) for guide in tedf['guide'].values]
    num_examples_te = len(tedf['guide'].values)

    outputs_te = np.zeros(num_examples_te) #random output

    other_single_value_inputs_te = np.empty((3, num_examples_te))
    #other_single_value_inputs_te[0, :] = tedf['linearfold_vals'].values/max(dataframe['linearfold_vals'].values) #normalize as the training data
    other_single_value_inputs_te[0, :] = tedf['linearfold_vals'].values/11.93 #normalize as the training data
    #other_single_value_inputs_te[1, :] = tedf['refseq_target_transcript_percent'].values
    other_single_value_inputs_te[1, :] = 1 # target isoform percent
    #other_single_value_inputs_te[2, :] = tedf['target unfold energy']/max(dataframe['target unfold energy'].values)
    other_single_value_inputs_te[2, :] = tedf['target unfold energy']/15.67


    all_cols_te = [
        encoded_guides_te,  # will be N x 4 from guide encoding
        other_single_value_inputs_te.T,
        outputs_te
    ]

    te = all_cols_te    

    te_out = te[-1]
    te = tuple(te[:-1])

    test_dataset = tf.data.Dataset.from_tensor_slices((te, te_out))

    # shuffle and batch
    test_dataset = prep_dataset(test_dataset, batch_size=128)

    return test_dataset


def CNN_all_features_input_dataset(testset_path):
    # test set data, read feature file path
    tedf = pd.read_csv(testset_path)
    encoded_guides_te = [one_hot_encode_sequence(guide) for guide in tedf['guide'].values]
    num_examples_te = len(tedf['guide'].values)

    outputs_te = np.zeros(num_examples_te) #random output

    other_single_value_inputs_te = np.empty((9, num_examples_te))
    #other_single_value_inputs_te[0, :] = tedf['linearfold_vals'].values/max(dataframe['linearfold_vals'].values) #normalize as the training data
    other_single_value_inputs_te[0, :] = tedf['linearfold_vals'].values/11.93 #normalize as the training data
    other_single_value_inputs_te[1, :] = tedf['is_5UTR'].values
    other_single_value_inputs_te[2, :] = tedf['is_CDS'].values
    other_single_value_inputs_te[3, :] = tedf['is_3UTR'].values
    other_single_value_inputs_te[4, :] = tedf['refseq_target_transcript_percent'].values # target isoform percent
    #other_single_value_inputs_te[2, :] = tedf['target unfold energy']/max(dataframe['target unfold energy'].values)
    other_single_value_inputs_te[5, :] = tedf['target unfold energy']/15.67
    other_single_value_inputs_te[6, :] = tedf['UTR5_position']
    other_single_value_inputs_te[7, :] = tedf['CDS_position']/0.99981488
    other_single_value_inputs_te[8, :] = tedf['UTR3_position']/0.99470448

    # 
    all_cols_te = [
        encoded_guides_te,  # will be N x 4 from guide encoding
        other_single_value_inputs_te.T,
        outputs_te
    ]


    te = all_cols_te    

    te_out = te[-1]
    te = tuple(te[:-1])

    test_dataset = tf.data.Dataset.from_tensor_slices((te, te_out))

    # shuffle and batch
    test_dataset = prep_dataset(test_dataset, batch_size=128)

    return test_dataset
