import pandas as pd
import numpy as np
import pdb
#from Bio.Seq import Seq
import sys
import csv
from itertools import chain
from typing import TextIO
import re

def parse_guide_linearfold_fasta_into_dict_contrafold(fname):
    fasta_file = open(fname)
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


def parse_target_flanks_linearfold_fasta_into_dict_contrafold(fname):
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


def parse_target_flanks_constraints_linearfold_fasta_into_dict_contrafold(fname):
    #flank_num = flank_len
    #fname = 'linearfold_output/linfold_guides_constrains_nearby'+str(flank_num)+'_output.txt'
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



def main(fpath):
    dataframe = pd.read_csv(fpath)
    #folder_name = '/'.join(fpath.split('/')[:-1])
    prefix = fpath[:-18]
    guide_linfold_c = prefix +'_linfold_guides_output.txt'
    target_flank_native = prefix +'_linfold_target_flank15_output.txt'
    target_flank_constraints = prefix +'_linfold_constraints_target_flank15_output.txt'

    num_examples = len(dataframe['guide'].values)

    #guide mfe energy
    lin_seq_dict, lin_result_dict = parse_guide_linearfold_fasta_into_dict_contrafold(guide_linfold_c)

    linearfold_vals = [lin_result_dict[guide] for guide in dataframe['guide'].values]
    for ii in range(num_examples):
        linearfold_vals[ii] = abs(linearfold_vals[ii]-6.48)

    #target with nearby seq, dg of native and unfolded
    flank_l = 15
    lin_seq_flanks_dict, lin_result_flanks_dict = parse_target_flanks_linearfold_fasta_into_dict_contrafold(target_flank_native)
    linearfold_vals_target = [lin_result_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_'+str(flank_l)].values] #native energy

    unfold_lin_seq_flanks_dict, unfold_lin_result_flanks_dict = parse_target_flanks_constraints_linearfold_fasta_into_dict_contrafold(target_flank_constraints)
    unfold_linearfold_vals_target = [unfold_lin_result_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_'+str(flank_l)].values] #unfolded target energy
    ddg = [] #energy required to unfold the guide binding region
    for jj in range(num_examples):
        ddg.append((linearfold_vals_target[jj]-unfold_linearfold_vals_target[jj]))
        
    dataframe['linearfold_vals']=linearfold_vals
    dataframe['target unfold energy']=ddg

    #select features
    df_select = dataframe[['transcript id','guide','linearfold_vals','target unfold energy']]
    df_select.to_csv(prefix+'_guides_integrated_features.csv')


if __name__ == '__main__':
    main(str(sys.argv[1]))
