from itertools import chain
from typing import TextIO

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf
from utils import base_positions, flip_dict, linearfold_positions
import numpy as np
import re
import sklearn
import pdb



#dataset_filtered_csv_path = 'dataset/integrated_guide_feature_filtered_new_ver3.csv' #new unchopped ratio
#gene_fasta_path = 'dataset/essential_transcripts.fasta'
#linearfold_fasta_path = 'dataset/guides_linearfold_v.txt'
#linearfold_fasta_path_c = 'dataset/guides_linearfold_c.txt'


genes = ['RPS14', 'CDC5L', 'POLR2I', 'RPS7', 'XAB2', 'RPS19BP1', 'RPL23A', 'SUPT6H', 'PRPF31', 'U2AF1', 'PSMD7',
         'Hsp10', 'RPS13', 'PHB', 'RPS9', 'EIF5B', 'RPS6', 'RPS11', 'SUPT5H', 'SNRPD2', 'RPL37', 'RPSA', 'COPS6',
         'DDX51', 'EIF4A3', 'KARS', 'RPL5', 'RPL32', 'SF3A1', 'RPS3A', 'SF3B3', 'POLR2D', 'RPS15A', 'RPL31', 'PRPF19',
         'SF3B2', 'RPS4X', 'CSE1L', 'RPL6', 'COPZ1', 'PSMB2', 'RPL7', 'PHB2', 'ARCN1', 'RPA2', 'NUP98', 'RPS3', 'EEF2',
         'USP39', 'PSMD1', 'NUP93', 'AQR', 'RPL34', 'PSMA1', 'RPS27A']

bprna_features = {'X', 'H', 'E', 'S', 'M', 'I', 'B'}

one_hot_gene_selection = dict(
    (gene, i) for i, gene in enumerate(genes))  # convert above to dictionary that will have 1 hot encoding


def parse_gene_fasta_into_dict():
    fasta_file = open(gene_fasta_path)
    gene_dict = {}
    cur_gene = None
    for line in fasta_file:
        if '(' in line:
            gene = line[line.find('(') + 1:line.find(')')]
            gene_dict[gene] = []
            cur_gene = gene
        else:
            for char in line:
                if char in base_positions.keys():
                    gene_dict[cur_gene].append(char)

    return gene_dict


def parse_guide_linearfold_fasta_into_dict():
    fasta_file = open(linearfold_fasta_path)
    seq_dict = {}
    score_dict = {}

    def parse_one_example(fasta: TextIO):
        descr_line = fasta.readline()
        if not descr_line:
            return None, None, None
        guide_seq = fasta.readline().strip()[36:]
        linearfold_and_result = fasta.readline()
        match = re.match('([\\.|\\(|\\)]+) \((\-?[0-9]*\.[0-9]+)\)', linearfold_and_result)
        linseq, score = match.groups()
        score = float(score)

        assert '>' in descr_line

        return guide_seq, linseq, score

    while True:
        key, seq, score = parse_one_example(fasta_file)
        if key is None:
            break
        seq_dict[key] = seq
        score_dict[key] = score

    fasta_file.close()

    return seq_dict, score_dict

def parse_guide_linearfold_fasta_into_dict_contrafold():
    fasta_file = open(linearfold_fasta_path_c)
    seq_dict = {}
    score_dict = {}

    def parse_one_example(fasta: TextIO):
        descr_line = fasta.readline()
        if not descr_line:
            return None, None, None
        guide_seq = fasta.readline().strip()[36:]
        linearfold_and_result = fasta.readline()
        match = re.match('([\\.|\\(|\\)]+) \((\-?[0-9]*\.[0-9]+)\)', linearfold_and_result)
        linseq, score = match.groups()
        score = float(score)

        assert '>' in descr_line

        return guide_seq, linseq, score

    while True:
        key, seq, score = parse_one_example(fasta_file)
        if key is None:
            break
        seq_dict[key] = seq
        score_dict[key] = score

    fasta_file.close()

    return seq_dict, score_dict


def parse_target_flanks_linearfold_fasta_into_dict_contrafold(flank_len = 15):
    flank_num = flank_len
    fname = 'dataset/linfold_guides_nearby'+str(flank_num)+'_output.txt'
    fasta_file = open(fname)
    seq_dict = {}
    score_dict = {}

    def parse_one_example(fasta: TextIO):
        descr_line = fasta.readline()
        if not descr_line:
            return None, None, None
        target_seq = fasta.readline().strip() #target with flanks
        linearfold_and_result = fasta.readline()
        match = re.match('([\\.|\\(|\\)]+) \((\-?[0-9]*\.[0-9]+)\)', linearfold_and_result)
        linseq, score = match.groups()
        score = float(score)

        assert '>' in descr_line

        return target_seq, linseq, score #return target seq with flanks

    while True:
        key, seq, score = parse_one_example(fasta_file)
        if key is None:
            break
        seq_dict[key] = seq
        score_dict[key] = score

    fasta_file.close()

    return seq_dict, score_dict


def parse_target_flanks_constraints_linearfold_fasta_into_dict_contrafold(flank_len = 15):
    flank_num = flank_len
    fname = 'dataset/linfold_guides_constrains_nearby'+str(flank_num)+'_output.txt'
    fasta_file = open(fname)
    seq_dict = {}
    score_dict = {}

    def parse_one_example(fasta: TextIO):
        descr_line = fasta.readline()
        if not descr_line:
            return None, None, None
        target_seq = fasta.readline().strip() #target with flanks
        constraints = fasta.readline()
        linearfold_and_result = fasta.readline()
        match = re.match('([\\.|\\(|\\)]+) \((\-?[0-9]*\.[0-9]+)\)', linearfold_and_result)
        linseq, score = match.groups()
        score = float(score)

        assert '>' in descr_line

        return target_seq, linseq, score #return target seq with flanks

    while True:
        key, seq, score = parse_one_example(fasta_file) #key is target seq with flanks
        if key is None:
            break
        seq_dict[key] = seq #linfold_seq
        score_dict[key] = score #linfold_score

    fasta_file.close()

    return seq_dict, score_dict


def one_hot_encode_linearfold(seq, remove_universal_start=False):
    if remove_universal_start:  # all linearfold guides have a constant 36 pair starter sequence
        seq = seq[36:]
    output_len = len(seq)
    encoded_seq = np.zeros((output_len, 3), dtype=np.float32)
    for i, fold in enumerate(seq):
        encoded_seq[i][linearfold_positions[fold]] = 1
    return encoded_seq


def one_hot_encode_linearfold_onechannel(seq, remove_universal_start=False):
    if remove_universal_start:  # all linearfold guides have a constant 36 pair starter sequence
        seq = seq[36:]
    output_len = len(seq)
    encoded_seq = np.zeros((output_len, 1), dtype=np.float32)
    for i, fold in enumerate(seq):
        if fold != '.': #pair with other bases
            encoded_seq[i,0] = 1
    return encoded_seq


def one_hot_encode_sequence(seq, pad_to_len=-1):
    output_len = len(seq)
    if pad_to_len > 0:
        assert pad_to_len >= output_len
        output_len = pad_to_len

    encoded_seq = np.zeros((output_len, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        encoded_seq[i][base_positions[base]] = 1
    return encoded_seq


def encoded_to_str(encoded_seq):
    indices = np.argmax(encoded_seq, axis=1)
    return ''.join([base_positions[i] for i in indices])


def complement_str(string):
    return ''.join([flip_dict[char] for char in string])


def complement_encoding(seq):
    encoded_seq = np.zeros((len(seq), 4))
    for i, base in enumerate(seq):
        encoded_seq[i][base_positions[flip_dict[base]]] = 1
    return encoded_seq


def reverse_complement_encoding(seq):
    return np.flip(complement_encoding(seq), axis=0)


def reverse_complement(encoding):
    return reverse_complement_encoding(encoded_to_str(encoding))


def get_gene_encodings(pad=True):
    gene_dict_of_strings = parse_gene_fasta_into_dict()

    pad_to_len = -1  # means do not pad
    if pad:
        pad_to_len = max(len(string) for string in gene_dict_of_strings.values())

    encoded_genes = {}
    for gene in gene_dict_of_strings.keys():
        encoded_genes[gene] = one_hot_encode_sequence(gene_dict_of_strings[gene], pad_to_len=pad_to_len)

    return encoded_genes


def create_gene_splits(gene_strings, values_to_split: list, seed=113):
    if seed:
        np.random.seed(seed)
    non_train_genes = np.random.choice(genes, 11, replace=False)
    val_genes = non_train_genes[:5]
    test_genes = non_train_genes[5:]

    print('val:', val_genes)
    print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids) - set(test_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val, test


def create_gene_splits_kfold(gene_strings, values_to_split: list, kfold, split):
    # use number [0, 1, 2, 3, 4] as index
    assert split >= 0 and split < kfold
    if kfold == 5:
        non_train_genes = genes[split * 11: (split + 1) * 11]
        val_genes = non_train_genes[:5]
        test_genes = non_train_genes[5:]
    elif kfold == 11:
        num_genes = len(genes)
        val_genes = genes[split * 5: (split + 1) * 5]
        if split != 10:
            test_genes = genes[((split + 1) * 5): (split + 2) * 5]
        else:
            test_genes = genes[0:5]
    print('val:', val_genes)
    print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids) - set(test_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val, test


def create_gene_splits_filter1_kfold(gene_strings, values_to_split: list, kfold, split):
    # use number [0, 1, 2, 3, 4] as index
    genes_filter_1 = ['RPS6', 'PRPF19', 'RPL34', 'Hsp10', 'POLR2I', 'EIF5B', 'RPL31',
       'RPS3A', 'CSE1L', 'XAB2', 'PSMD7', 'SUPT6H', 'EEF2', 'RPS11',
       'SNRPD2', 'RPL37', 'SF3B3', 'DDX51', 'RPL7', 'RPS9', 'KARS',
       'SF3A1', 'RPL32', 'PSMB2', 'RPS7', 'EIF4A3', 'U2AF1', 'PSMA1',
       'PHB', 'POLR2D', 'RPSA', 'RPL23A', 'NUP93', 'AQR', 'RPA2',
       'SUPT5H', 'RPL6', 'RPS13', 'SF3B2', 'RPS27A', 'PRPF31', 'COPZ1',
       'RPS4X', 'PSMD1', 'RPS14', 'NUP98', 'USP39', 'CDC5L', 'RPL5',
       'PHB2', 'RPS15A', 'RPS3', 'ARCN1', 'COPS6']
    assert split >= 0 and split < kfold
    if kfold == 9:
        val_genes = genes_filter_1[split * 6: (split + 1) * 6]
        if split != 8:
            test_genes = genes_filter_1[((split + 1) * 6): (split + 2) * 6]
        else:
            test_genes = genes_filter_1[0:6]
    print('val:', val_genes)
    print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids) - set(test_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val, test

def create_gene_splits_filter1_new_kfold(gene_strings, values_to_split: list, kfold, split):
    # use number [0, 1, 2, 3, 4] as index
    genes_filter_1_new = ['RPS6', 'PRPF19', 'RPL34', 'Hsp10', 'POLR2I', 'DDX51', 'RPL31',
       'RPS3A', 'CSE1L', 'XAB2', 'PSMD7', 'EIF5B', 'EEF2', 'RPS11',
       'SNRPD2', 'RPL37', 'SF3B3', 'SUPT6H', 'RPL7', 'RPS9', 'KARS',
       'SF3A1', 'RPL32', 'PSMB2', 'RPS7', 'EIF4A3', 'U2AF1', 'PSMA1',
       'PHB', 'POLR2D', 'RPSA', 'RPL23A', 'NUP93', 'AQR', 'RPA2',
       'SUPT5H', 'RPL6', 'RPS13', 'SF3B2', 'RPS27A', 'PRPF31', 'COPZ1',
       'RPS4X', 'PSMD1', 'RPS14', 'NUP98', 'USP39', 'COPS6', 'RPL5',
       'PHB2', 'RPS15A', 'RPS3', 'ARCN1', 'CDC5L']
    assert split >= 0 and split < kfold
    if kfold == 9:
        val_genes = genes_filter_1_new[split * 6: (split + 1) * 6]
        if split != 8:
            test_genes = genes_filter_1_new[((split + 1) * 6): (split + 2) * 6]
        else:
            test_genes = genes_filter_1_new[0:6]
    print('val:', val_genes)
    print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids) - set(test_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val, test

def create_gene_splits_filter2_kfold(gene_strings, values_to_split: list, kfold, split):
    # use number [0, 1, 2, 3, 4] as index
    genes_filter_2 = ['COPZ1', 'RPS27A', 'PSMA1', 'SNRPD2', 'EIF4A3', 'RPS4X', 'ARCN1',
       'POLR2I', 'SF3B3', 'RPS15A', 'RPL6', 'RPS9', 'EIF5B', 'RPA2',
       'XAB2', 'NUP93', 'RPS11', 'RPL7', 'SUPT6H', 'PHB', 'Hsp10',
       'U2AF1', 'RPL5', 'RPS7', 'PSMB2', 'USP39', 'AQR', 'RPS14',
       'RPL23A', 'RPSA', 'SUPT5H', 'RPL32', 'NUP98', 'PRPF19', 'PSMD1',
       'RPS13', 'COPS6', 'SF3A1', 'RPS3', 'KARS', 'PHB2', 'RPL31',
       'DDX51', 'PSMD7', 'RPL34', 'CSE1L', 'EEF2', 'RPS3A', 'POLR2D',
       'PRPF31', 'RPL37', 'SF3B2', 'RPS6']
    assert split >= 0 and split < kfold
    if kfold == 9:
        if split != 8:
            val_genes = genes_filter_2[split * 6: (split + 1) * 6]
            test_genes = genes_filter_2[((split + 1) * 6): min((split + 2) * 6,53)]
        else:
            val_genes = genes_filter_2[split * 6: 53]
            test_genes = genes_filter_2[0:6]
    print('val:', val_genes)
    print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids) - set(test_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val, test


def create_gene_splits_filter5_kfold(gene_strings, values_to_split: list, kfold, split):
    # use number [0, 1, 2, 3, 4] as index
    genes_filter_5 = ['RPS6', 'SF3B2', 'RPL34', 'USP39', 'PSMD7', 'RPL31', 'RPS9',
       'KARS', 'NUP98', 'POLR2D', 'RPS4X', 'PRPF19', 'RPS3', 'SF3A1',
       'SUPT5H', 'EEF2', 'RPS13', 'NUP93', 'PHB', 'RPL32', 'RPL7',
       'RPS15A', 'AQR', 'PSMA1', 'DDX51', 'RPL5', 'RPS3A', 'CSE1L',
       'PRPF31', 'POLR2I', 'RPS7', 'PSMD1', 'SNRPD2', 'XAB2', 'RPA2',
       'RPSA', 'RPL23A', 'Hsp10', 'RPS27A', 'SUPT6H', 'RPL6', 'RPS11',
       'U2AF1', 'SF3B3', 'EIF5B', 'PHB2', 'EIF4A3', 'RPS14', 'RPL37',
       'ARCN1']
    assert split >= 0 and split < kfold
    if kfold == 10:
        val_genes = genes_filter_5[split * 5: (split + 1) * 5]
        if split != 9:
            test_genes = genes_filter_5[((split + 1) * 5): (split + 2) * 5]
        else:
            test_genes = genes_filter_5[0:5]

    print('val:', val_genes)
    print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids) - set(test_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val, test


def create_gene_splits_no_test_kfold(gene_strings, values_to_split: list, kfold, split):
    assert split >= 0 and split < kfold
    if kfold == 5:
        non_train_genes = genes[split * 11: (split + 1) * 11]
        val_genes = non_train_genes[:5]
        test_genes = non_train_genes[5:]
    elif kfold == 11:
        num_genes = len(genes)
        val_genes = genes[split * 5: (split + 1) * 5]
#        if split != 10:
#            test_genes = genes[((split + 1) * 5): (split + 2) * 5]
#        else:
#            test_genes = genes[0:5]
    print('val:', val_genes)
#    print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
 #   test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
 #   train_ids = list((set(range(len(gene_strings))) - set(val_ids) - set(test_ids)))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
#    test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val

def create_gene_splits_filter1_no_test_kfold(gene_strings, values_to_split: list, kfold, split):
    genes_filter_1 = ['RPS6', 'PRPF19', 'RPL34', 'Hsp10', 'POLR2I', 'EIF5B', 'RPL31',
       'RPS3A', 'CSE1L', 'XAB2', 'PSMD7', 'SUPT6H', 'EEF2', 'RPS11',
       'SNRPD2', 'RPL37', 'SF3B3', 'DDX51', 'RPL7', 'RPS9', 'KARS',
       'SF3A1', 'RPL32', 'PSMB2', 'RPS7', 'EIF4A3', 'U2AF1', 'PSMA1',
       'PHB', 'POLR2D', 'RPSA', 'RPL23A', 'NUP93', 'AQR', 'RPA2',
       'SUPT5H', 'RPL6', 'RPS13', 'SF3B2', 'RPS27A', 'PRPF31', 'COPZ1',
       'RPS4X', 'PSMD1', 'RPS14', 'NUP98', 'USP39', 'CDC5L', 'RPL5',
       'PHB2', 'RPS15A', 'RPS3', 'ARCN1', 'COPS6']
    assert split >= 0 and split < kfold
    if kfold == 9:
        val_genes = genes_filter_1[split * 6: (split + 1) * 6]
        #if split != 8:
        #    test_genes = genes_filter_1[((split + 1) * 6): (split + 2) * 6]
        #else:
        #    test_genes = genes_filter_1[0:6]
    print('val:', val_genes)
    #print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    #test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    #test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val

def create_gene_splits_filter1_test_asval_kfold(gene_strings, values_to_split: list, kfold, split):
    genes_filter_1 = ['RPS6', 'PRPF19', 'RPL34', 'Hsp10', 'POLR2I', 'EIF5B', 'RPL31',
       'RPS3A', 'CSE1L', 'XAB2', 'PSMD7', 'SUPT6H', 'EEF2', 'RPS11',
       'SNRPD2', 'RPL37', 'SF3B3', 'DDX51', 'RPL7', 'RPS9', 'KARS',
       'SF3A1', 'RPL32', 'PSMB2', 'RPS7', 'EIF4A3', 'U2AF1', 'PSMA1',
       'PHB', 'POLR2D', 'RPSA', 'RPL23A', 'NUP93', 'AQR', 'RPA2',
       'SUPT5H', 'RPL6', 'RPS13', 'SF3B2', 'RPS27A', 'PRPF31', 'COPZ1',
       'RPS4X', 'PSMD1', 'RPS14', 'NUP98', 'USP39', 'CDC5L', 'RPL5',
       'PHB2', 'RPS15A', 'RPS3', 'ARCN1', 'COPS6']
    assert split >= 0 and split < kfold
    if kfold == 9:
        val_genes = genes_filter_1[split * 6: (split + 1) * 6]
        if split != 8:
            test_genes = genes_filter_1[((split + 1) * 6): (split + 2) * 6]
        else:
            test_genes = genes_filter_1[0:6]
        val_genes = val_genes + test_genes
    print('val:', val_genes)
    #print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    #test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    #test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val


def create_gene_splits_filter2_no_test_kfold(gene_strings, values_to_split: list, kfold, split):
    genes_filter_2 = ['COPZ1', 'RPS27A', 'PSMA1', 'SNRPD2', 'EIF4A3', 'RPS4X', 'ARCN1',
       'POLR2I', 'SF3B3', 'RPS15A', 'RPL6', 'RPS9', 'EIF5B', 'RPA2',
       'XAB2', 'NUP93', 'RPS11', 'RPL7', 'SUPT6H', 'PHB', 'Hsp10',
       'U2AF1', 'RPL5', 'RPS7', 'PSMB2', 'USP39', 'AQR', 'RPS14',
       'RPL23A', 'RPSA', 'SUPT5H', 'RPL32', 'NUP98', 'PRPF19', 'PSMD1',
       'RPS13', 'COPS6', 'SF3A1', 'RPS3', 'KARS', 'PHB2', 'RPL31',
       'DDX51', 'PSMD7', 'RPL34', 'CSE1L', 'EEF2', 'RPS3A', 'POLR2D',
       'PRPF31', 'RPL37', 'SF3B2', 'RPS6']
    assert split >= 0 and split < kfold
    if kfold == 9:
        if split != 8:
            val_genes = genes_filter_2[split * 6: (split + 1) * 6]
            #test_genes = genes_filter_2[((split + 1) * 6): min((split + 2) * 6,53)]
        else:
            val_genes = genes_filter_2[split * 6: 53]
            #test_genes = genes_filter_2[0:6]
    print('val:', val_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    #test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    #test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val


def create_gene_splits_filter2_test_asval_kfold(gene_strings, values_to_split: list, kfold, split):
    # use number [0, 1, 2, 3, 4] as index
    genes_filter_2 = ['COPZ1', 'RPS27A', 'PSMA1', 'SNRPD2', 'EIF4A3', 'RPS4X', 'ARCN1',
       'POLR2I', 'SF3B3', 'RPS15A', 'RPL6', 'RPS9', 'EIF5B', 'RPA2',
       'XAB2', 'NUP93', 'RPS11', 'RPL7', 'SUPT6H', 'PHB', 'Hsp10',
       'U2AF1', 'RPL5', 'RPS7', 'PSMB2', 'USP39', 'AQR', 'RPS14',
       'RPL23A', 'RPSA', 'SUPT5H', 'RPL32', 'NUP98', 'PRPF19', 'PSMD1',
       'RPS13', 'COPS6', 'SF3A1', 'RPS3', 'KARS', 'PHB2', 'RPL31',
       'DDX51', 'PSMD7', 'RPL34', 'CSE1L', 'EEF2', 'RPS3A', 'POLR2D',
       'PRPF31', 'RPL37', 'SF3B2', 'RPS6']
    assert split >= 0 and split < kfold
    if kfold == 9:
        if split != 8:
            val_genes = genes_filter_2[split * 6: (split + 1) * 6]
            test_genes = genes_filter_2[((split + 1) * 6): min((split + 2) * 6,53)]
        else:
            val_genes = genes_filter_2[split * 6: 53]
            test_genes = genes_filter_2[0:6]

        val_genes = val_genes + test_genes
    print('val:', val_genes)
    #print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    #test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    #test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val


def create_gene_splits_filter5_test_asval_kfold(gene_strings, values_to_split: list, kfold, split):
    # use number [0, 1, 2, 3, 4] as index
    genes_filter_5 = ['RPS6', 'SF3B2', 'RPL34', 'USP39', 'PSMD7', 'RPL31', 'RPS9',
       'KARS', 'NUP98', 'POLR2D', 'RPS4X', 'PRPF19', 'RPS3', 'SF3A1',
       'SUPT5H', 'EEF2', 'RPS13', 'NUP93', 'PHB', 'RPL32', 'RPL7',
       'RPS15A', 'AQR', 'PSMA1', 'DDX51', 'RPL5', 'RPS3A', 'CSE1L',
       'PRPF31', 'POLR2I', 'RPS7', 'PSMD1', 'SNRPD2', 'XAB2', 'RPA2',
       'RPSA', 'RPL23A', 'Hsp10', 'RPS27A', 'SUPT6H', 'RPL6', 'RPS11',
       'U2AF1', 'SF3B3', 'EIF5B', 'PHB2', 'EIF4A3', 'RPS14', 'RPL37',
       'ARCN1']
    assert split >= 0 and split < kfold
    if kfold == 10:
        val_genes = genes_filter_5[split * 5: (split + 1) * 5]
        if split != 9:
            test_genes = genes_filter_5[((split + 1) * 5): (split + 2) * 5]
        else:
            test_genes = genes_filter_5[0:5]

        val_genes = val_genes + test_genes
    print('val:', val_genes)
    #print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    #test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    #test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val


def create_gene_splits_filter5_new_test_asval_kfold(gene_strings, values_to_split: list, kfold, split):
    # use number [0, 1, 2, 3, 4] as index
    genes_filter_5_new = ['EIF5B', 'RPL37', 'CSE1L', 'PSMD1', 'RPS13', 'ARCN1', 'SF3A1',
       'RPL34', 'RPS11', 'EEF2', 'RPA2', 'PHB', 'KARS', 'RPS3A', 'RPS4X',
       'SUPT6H', 'PSMA1', 'RPS14', 'EIF4A3', 'RPL5', 'POLR2I', 'SF3B3',
       'AQR', 'RPS9', 'RPL31', 'PSMD7', 'RPL32', 'Hsp10', 'PRPF19',
       'RPSA', 'SUPT5H', 'XAB2', 'RPS3', 'PHB2', 'RPL6', 'PRPF31',
       'USP39', 'NUP93', 'RPL23A', 'RPS7', 'DDX51', 'NUP98', 'U2AF1',
       'SF3B2', 'RPL7', 'POLR2D', 'RPS27A', 'SNRPD2', 'RPS15A', 'RPS6']
    assert split >= 0 and split < kfold
    if kfold == 10:
        val_genes = genes_filter_5_new[split * 5: (split + 1) * 5]
        if split != 9:
            test_genes = genes_filter_5_new[((split + 1) * 5): (split + 2) * 5]
        else:
            test_genes = genes_filter_5_new[0:5]

        val_genes = val_genes + test_genes
    print('val:', val_genes)
    #print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    #test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    #test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val


def create_gene_splits_filter6_test_asval_kfold(gene_strings, values_to_split: list, kfold, split):
    # use number [0, 1, 2, 3, 4] as index
    genes_filter_6 = ['SUPT6H', 'Hsp10', 'RPS14', 'PSMD1', 'RPL31', 'RPA2', 'NUP98',
       'RPL34', 'RPS7', 'EEF2', 'ARCN1', 'SF3A1', 'RPS15A', 'PRPF19',
       'PHB2', 'SUPT5H', 'PSMA1', 'RPS3', 'RPL23A', 'RPSA', 'POLR2I',
       'RPS27A', 'NUP93', 'RPS13', 'RPL6', 'DDX51', 'PHB', 'SNRPD2',
       'RPS9', 'RPL7', 'PSMD7', 'USP39', 'KARS', 'SF3B2', 'RPS6',
       'POLR2D', 'RPL37', 'CSE1L', 'EIF4A3', 'RPS3A', 'PRPF31', 'SF3B3',
       'AQR', 'RPS11', 'RPL5', 'RPL32', 'XAB2', 'U2AF1', 'RPS4X']
    assert split >= 0 and split < kfold

    if kfold == 10:
        if split != 9:
            val_genes = genes_filter_6[split * 5: (split + 1) * 5]
            test_genes = genes_filter_6[((split + 1) * 5): min((split + 2) * 5,49)]
        else:
            val_genes = genes_filter_6[split * 5:]
            test_genes = genes_filter_6[0:5]

        val_genes = val_genes + test_genes
    print('val:', val_genes)
    #print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    #test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    #test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val


def create_gene_splits_no_test_nbt_val_kfold(gene_strings, values_to_split: list, kfold, split):
    assert split >= 0 and split < kfold
    if kfold == 5:
        non_train_genes = genes[split * 11: (split + 1) * 11]
        val_genes = non_train_genes[:5]
        test_genes = non_train_genes[5:]
    elif kfold == 11:
        num_genes = len(genes)
        val_genes = genes[split * 5: (split + 1) * 5]+['CD46','CD55','CD71']

    print('val:', val_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))

    train_ids = list((set(range(len(gene_strings))) - set(val_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]

    return train, val

def create_gene_splits_no_test_cd_val_kfold(gene_strings, values_to_split: list, kfold, split):
    assert split >= 0 and split < kfold
    if kfold == 5:
        non_train_genes = genes[split * 11: (split + 1) * 11]
        val_genes = non_train_genes[:5]
        test_genes = non_train_genes[5:]
    elif kfold == 11:
        num_genes = len(genes)
        val_genes = genes[split * 5: (split + 1) * 5]+['CD58','CD81']

    print('val:', val_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))

    train_ids = list((set(range(len(gene_strings))) - set(val_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]

    return train, val



def create_dataset_inputs_and_outputs(input_list, outputs, classes):
    num_inputs = len(input_list)
    split_data = train_test_split(*input_list, outputs, classes, test_size=.1, stratify=classes)
    remaining_classes = split_data[-2]
    val_outputs = split_data[-3]
    remaining_outputs = split_data[-4]
    val_inputs = tuple(split_data[1:-4:2])
    remaining_inputs = tuple(split_data[:-4:2])

    split_data = train_test_split(*remaining_inputs, remaining_outputs, test_size=.1, stratify=remaining_classes)
    test_outputs = split_data[-1]
    train_outputs = split_data[-2]
    test_inputs = tuple(split_data[1:-2:2])
    train_inputs = tuple(split_data[:-2:2])

    return (train_inputs, train_outputs), (val_inputs, val_outputs), (test_inputs, test_outputs)


def create_guide_splits_kfold(values_to_split: list, kfold, split):
    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
    ss = 0
    for train_index, test_index in kf.split(values_to_split[-1]):
        if ss == int(split):
            #print("TRAIN:", train_index, "TEST:", test_index)
            val_ids = np.random.choice(train_index, size=len(test_index), replace=False)
            test_ids = test_index
            train_ids = list(set(train_index) - set(val_ids))
            #X_train, X_test = np.array(all_cols[0])[train_index], np.array(all_cols[0])[test_index]
            #y_train, y_test = np.array(all_cols[1])[train_index], np.array(all_cols[1])[test_index]
            train = [[arr[i] for i in train_ids] for arr in values_to_split]
            val = [[arr[i] for i in val_ids] for arr in values_to_split]
            test = [[arr[i] for i in test_ids] for arr in values_to_split]
            return train, val, test
        else:
            ss += 1




def prep_dataset(d: tf.data.Dataset, batch_size=16):
    d = d.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return d


if __name__ == '__main__':
    encodings = get_gene_encodings(pad=False)
    lens = dict([(gene, len(encoding)) for gene, encoding in encodings.items()])
    print(lens)


# pad right
def get_nearby_encoding(guide_encoding, guide_loc, gene_encoding, num_bases_either_side=10, pad=True, rcomp=False,
                        flag_dim=False):
    n = num_bases_either_side
    gene_len = len(gene_encoding)
    guide_len = len(guide_encoding)
    guide_index = int(np.round(guide_loc * gene_len))
    if abs(guide_index - guide_loc * gene_len) > 1e-2:
        print("rounded %f to %d" % (guide_loc * gene_len, guide_index))

    reverse_comp = reverse_complement(guide_encoding)
    assert np.all(reverse_comp == gene_encoding[guide_index:guide_index + guide_len])

    min_index = max(guide_index - n, 0)
    max_index = min(guide_index + guide_len + n, gene_len)
    gene_out = gene_encoding[min_index:max_index]

    expected_len = guide_len + num_bases_either_side * 2

    if pad and len(gene_out) < expected_len:
        padded = np.zeros((expected_len, 4))
        padded[0:len(gene_out), :] = gene_out
        gene_out = padded

    if rcomp:
        gene_out = reverse_complement(gene_out)

    if flag_dim:
        new_out = np.zeros((gene_out.shape[0], 5))
        new_out[:, :4] = gene_out
        new_out[num_bases_either_side:-num_bases_either_side, 4] = 1
        gene_out = new_out

    return gene_out

def get_nearby_encoding_rv(guide_encoding, guide_loc, gene_encoding, num_bases_either_side=3, left = True, right = True, pad=True):
    n = num_bases_either_side
    gene_len = len(gene_encoding)
    guide_len = len(guide_encoding)
    guide_index = int(guide_loc)

    #flank seq on both sides
    min_index = max(guide_index - n, 0)
    max_index = min(guide_index + guide_len + n, gene_len)
    expected_len = guide_len + num_bases_either_side * 2
    if left == False: #only add right flanking seq from the gene
        min_index = guide_index
        expected_len = guide_len + num_bases_either_side
    if right == False:
        max_index = guide_index + guide_len
        expected_len = guide_len + num_bases_either_side

    gene_out = gene_encoding[min_index:max_index]
    guide_flanks_encode = reverse_complement(gene_out)

    if pad and (len(gene_out) < expected_len):
        padded = np.zeros((expected_len, 4))
        if max_index == gene_len: #end of gene
            #padded[0:len(gene_out), :] = gene_out
            padded[-len(gene_out):, :] = guide_flanks_encode
        else:
            #padded[-len(gene_out):, :] = gene_out
            padded[0:len(gene_out), :] = guide_flanks_encode
        guide_flanks_encode = padded

    return guide_flanks_encode


def normalize(a: np.ndarray):
    """
    :param a: numpy array of size N x D, where N is number of examples, D is number of features
    :return: a, normalized so that all feature columns are now between 0 and 1
    """
    a_normed, norms = sklearn.preprocessing.normalize(a, norm='max', axis=0, return_norm=True)
    print("Norms:", norms)
    return a_normed
