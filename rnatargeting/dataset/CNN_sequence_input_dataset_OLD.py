import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

from rnatargeting.dataset.utils import *


def CNN_sequence_input_dataset(args):
    #dataframe = pd.read_csv('dataset/integrated_guide_feature_filtered_f24_mismatch3_all_features.csv')
    #num_examples = len(dataframe['gene'].values)

    #lin_seq_dict, lin_result_dict = parse_guide_linearfold_fasta_into_dict_contrafold()

    #encoded_guides = [one_hot_encode_sequence(guide) for guide in dataframe['guide'].values]
    #encoded_linearfold = [one_hot_encode_linearfold(lin_seq_dict[guide], remove_universal_start=True) for guide in
    #                      dataframe['guide'].values]

                          
    #linearfold_vals = [lin_result_dict[guide] for guide in dataframe['guide'].values]
    #for ii in range(num_examples):
    #    linearfold_vals[ii] = abs(linearfold_vals[ii]-6.48)


    #target with nearby seq, dg of native and unfolded
    #flank_l = int(args.flanklength)
    #lin_seq_flanks_dict, lin_result_flanks_dict = parse_target_flanks_linearfold_fasta_into_dict_contrafold(flank_len = flank_l)
    #linearfold_vals_target = [lin_result_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_'+str(flank_l)].values] #native energy
    #lin_seq_flanks = [lin_seq_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_100'].values]
    #unfold_lin_seq_flanks_dict, unfold_lin_result_flanks_dict = parse_target_flanks_constraints_linearfold_fasta_into_dict_contrafold(flank_len = flank_l)
    #unfold_linearfold_vals_target = [unfold_lin_result_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_'+str(flank_l)].values] #unfolded target energy
    #ddg = [] #energy required to unfold the guide binding region
    #for jj in range(num_examples):
    #    ddg.append((linearfold_vals_target[jj]-unfold_linearfold_vals_target[jj]))


    #other_single_value_inputs = np.empty((3, num_examples))
    #other_single_value_inputs[0, :] = dataframe['linearfold_vals'].values
    #other_single_value_inputs[1, :] = dataframe['refseq_target_transcript_percent'].values
    #other_single_value_inputs[2, :] = dataframe['target unfold energy'].values

    #classes = dataframe['binary_relative_ratio_075f'].values

    #dataframe['relative_ratio_filtered']= dataframe.groupby("gene")["ratio"].rank(pct=True)
    #outputs = dataframe['relative_ratio'].values if args.regression else classes.astype(np.float32)
    #outputs = outputs.tolist()

    #all_cols = [encoded_guides,  # will be N x 4 from guide encoding
    #            normalize(other_single_value_inputs.T),
                # classes,
    #            outputs
    #            ]

    #tr = all_cols

    #if args.kfold == None:
    #    tr, val, te_train = create_gene_splits(dataframe['gene'].values, all_cols)
    #else:
        #tr, val = create_gene_splits_no_test_kfold(dataframe['gene'].values, all_cols, args.kfold, args.split)
        #tr, val, te_train = create_gene_splits_kfold(dataframe['gene'].values, all_cols, args.kfold, args.split)
        #tr, val, te_train = create_gene_splits_filter1_kfold(dataframe['gene'].values, all_cols, args.kfold, args.split)
    #    tr, val = create_gene_splits_filter1_test_asval_kfold(dataframe['gene'].values, all_cols, args.kfold, args.split)


    # test set data, read feature file path
    tedf = pd.read_csv(args.testset_path)
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


    all_cols_te = [encoded_guides_te,  # will be N x 4 from guide encoding
                other_single_value_inputs_te.T,
                outputs_te
                ]


    te = all_cols_te    


    #tr_out = tr[-1]
    #tr = tuple(tr[:-1])
    #val_out = val[-1]
    #val = tuple(val[:-1])
    te_out = te[-1]
    te = tuple(te[:-1])

    #train_dataset = tf.data.Dataset.from_tensor_slices((tr, tr_out))
    #val_dataset = tf.data.Dataset.from_tensor_slices((val, val_out))
    test_dataset = tf.data.Dataset.from_tensor_slices((te, te_out))

    # shuffle and batch
    #train_dataset = prep_dataset(train_dataset, batch_size=128)
    #val_dataset = prep_dataset(val_dataset, batch_size=128)
    test_dataset = prep_dataset(test_dataset, batch_size=128)

    return test_dataset
