import os
import re
import sys
import csv
import subprocess
from typing import TextIO
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


def run_linearfold(infile, outfile, params = []):
    sys.stderr.write(f"Running LinearFold on {infile}\n")
    with open(infile) as inF, open(outfile, 'w') as outF:
        subprocess.run(['LinearFold/linearfold'] + params, stdin=inF, stdout=outF)

def make_guide_library_features(fpath, tmpdir):
    sys.stderr.write(f"Making guide library features for {fpath}\n")

    # I/O
    input_fasta = fpath
    outfile_prefix = os.path.join(
        tmpdir, os.path.basename(os.path.splitext(fpath)[0])
    )
    output_gl = outfile_prefix + '_guide_library.csv'

    # handle multiple fasta sequences, support different genes (but don't put different isoform sequences for the same gene!)
    guide_dic = {} #guide sequence and features
    regex_nuc = re.compile(r'^[atgcuATGCU]+$')
    for record in SeqIO.parse(input_fasta, "fasta"):
        # parse seq info
        sequence = record.seq
        transcript_ID = record.id
        
        # check sequence format
        if not regex_nuc.match(str(sequence)):
            raise ValueError(
                f"Sequence {transcript_ID} contains non-nucleotide characters"
            )

        # design guides
        gene_guide_dic = {} #gene-specific guide dic
        trans_seq = sequence
        for i in range(0,len(trans_seq)-29):
            # Set target seq and spacer seq
            target = trans_seq[i:i+30]
            spacer = target.reverse_complement()

            # Replace U and upper
            spacer = (str(spacer)).upper()
            spacer = spacer.replace("U", "T")

            # Set flanking seq
            if (i != 0) and (i != len(trans_seq)-30):
                left_seq_15 = trans_seq[max(0,(i-15)):i]
                right_seq_15 = trans_seq[i+30:min((i+45),len(trans_seq))]
            elif i == 0:
                left_seq_15 = ''
                right_seq_15 = trans_seq[i+30:min((i+45),len(trans_seq))]
            elif i == len(trans_seq)-30:  
                left_seq_15 = trans_seq[max(0,(i-15)):i]
                right_seq_15 = ''
            nearby_seq_all_15 = left_seq_15 + target + right_seq_15 #target seq+flanks

            # Replace U and make uppercase
            nearby_seq_all_15 = (str(nearby_seq_all_15)).upper()
            nearby_seq_all_15 = nearby_seq_all_15.replace("U", "T")
            target = (str(target)).upper()
            target = target.replace("U", "T")

            # Add to gene_guide_dic
            if str(spacer) not in gene_guide_dic.keys(): 
                # first occurrance of guide
                gene_guide_dic[str(spacer)] = [
                        str(transcript_ID), str(spacer), [i], 1
                    ] + [
                        str(target), str(nearby_seq_all_15)
                    ]
            else: 
                # Another guide targeting the same transcript
                gene_guide_dic[str(spacer)][2].append(i)
                gene_guide_dic[str(spacer)][3] += 1
        
        # Add to guide dict
        for guide in gene_guide_dic.keys():
            if guide not in guide_dic.keys():
                guide_dic[guide] = gene_guide_dic[guide]
            else: #guide target other genes
                guide_dic.pop(guide) #remove the guide with off-targets
    
    # Check that guide library is not empty
    if len(guide_dic) == 0:
        raise ValueError(
            "No valid nucleotide sequences found in the fasta file!"
        )

    # Write to csv
    with open(output_gl, 'w') as outF:
        writer = csv.writer(outF)
        writer.writerow([
            'transcript id', 'guide', 'target_pos_list', 
            'target pos num', 'target_seq','nearby_seq_all_15'
        ])
        for g in guide_dic.keys():
            writer.writerow(guide_dic[g])

    # Make Linearfold input files
    ## Guide mfe input
    standard_prefix = "CAAGTAAACCCCTACCAACTGGTCGGGGTTTGAAAC"
    df = pd.read_csv(output_gl)
    num_examples = len(df['guide'].values)
    guideseq = df['guide'].values  
    ## Write to file
    outfile_gi = outfile_prefix + '_linfold_guides_input.fasta'
    with open(outfile_gi, 'w') as outF:
        for i in range(num_examples):
            outF.write(">guide #" + str(i + 1) + ", " +  df.iloc[i, 0] + "\n")
            outF.write(standard_prefix + guideseq[i] + "\n")

    # Target with flank 15 nt
    ## Native
    nearby_seq_all = df['nearby_seq_all_15'].values
    outfile_nb = outfile_prefix + "_linfold_guides_nearby15_input.fasta"
    with open(outfile_nb, 'w') as outF:
        for i in range(num_examples):
            outF.write(">guide #" + str(i + 1) + ", " +  df.iloc[i, 0] + "\n")
            outF.write(nearby_seq_all[i] + "\n")
            
    ## With constraints
    guide_rv = df['target_seq'].values
    outfile_nbc = outfile_prefix + "_linfold_guides_constraints_nearby15_input.fasta"
    with open(outfile_nbc, 'w') as outF:
        for i in range(num_examples):
            outF.write(">guide #" + str(i + 1) + ", " +  df.iloc[i, 0] + "\n")
            outF.write(nearby_seq_all[i] + "\n")
            target_index = nearby_seq_all[i].find(guide_rv[i])
            constraints = "?" * target_index + "." * 30 + "?" * (len(nearby_seq_all[i]) - 30 - target_index)
            outF.write(constraints + "\n")
    
    # Return output file paths
    return {
        'guide_library' : output_gl,
        'guide_input' : outfile_gi,
        'target_flank_input' : outfile_nb,
        'target_flank_c_input' : outfile_nbc
    }


def parse_guide_linearfold(fname, guide_seq=False):
    regex1 = re.compile(r'^[ATGCU]+$')
    regex2 = re.compile(r'.+ \(-?[0-9]+\.[0-9]+\)')
    seq_dict = {}
    score_dict = {}
    target_seq = None
    with open(fname) as inF:
        for line in inF:
            line = line.strip()
            if line.startswith('Unrecognized sequence:') or line == '':
                continue
            if regex1.match(line):
                # target-seq
                seq_len = 36 if guide_seq else 0
                target_seq = line[seq_len:]
            elif target_seq is not None and regex2.match(line):
                # lin-seq
                linseq,score = line.split(' ')
                score = float(score.strip('()'))
                score_dict[target_seq] = score   
    # return linearfold scores              
    return score_dict

def filter_result_dict(result_dict, values):
    results = []
    for value in values:
        try:
            results.append(result_dict[value])
        except KeyError:
            raise KeyError(f"Key {value} not found in result dict")
    return results

def linearfold_integrate_results(guide_library,
                                 guide_linfold_c, 
                                 target_flank_native, 
                                 target_flank_constraints,
                                 output_prefix):
    sys.stderr.write(f"Integrating Linearfold results...\n")

    # read guide library
    dataframe = pd.read_csv(guide_library)
    num_examples = len(dataframe['guide'].values)

    #guide mfe energy
    lin_result_dict = parse_guide_linearfold(guide_linfold_c, guide_seq=True)
    linearfold_vals = filter_result_dict(lin_result_dict, dataframe['guide'].values)
    for ii in range(num_examples):
        linearfold_vals[ii] = abs(linearfold_vals[ii] - 6.48)

    #target with nearby seq, dg of native and unfolded
    lin_result_flanks_dict = parse_guide_linearfold(target_flank_native)
    linearfold_vals_target = filter_result_dict(lin_result_flanks_dict, dataframe['nearby_seq_all_15'].values)
    
    #target with nearby seq, dg of native and unfolded
    unfold_lin_result_flanks_dict = parse_guide_linearfold(target_flank_constraints)
    unfold_linearfold_vals_target = filter_result_dict(unfold_lin_result_flanks_dict, dataframe['nearby_seq_all_15'].values)
    ddg = [] #energy required to unfold the guide binding region
    for jj in range(num_examples):
        ddg.append((linearfold_vals_target[jj] - unfold_linearfold_vals_target[jj]))

    # Add columns to dataframe
    dataframe['linearfold_vals'] = linearfold_vals
    dataframe['target unfold energy'] = ddg

    # write to csv
    df_select = dataframe[['transcript id','guide','linearfold_vals','target unfold energy']]
    outfile = output_prefix + '_guides_integrated_features.csv'
    df_select.to_csv(outfile)
    sys.stderr.write(f"  File written: {outfile}\n")

    # return output file path
    return outfile