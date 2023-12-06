import pandas as pd
import numpy as np
import csv
import sys
import os
from Bio import SeqIO
from Bio.Seq import Seq


def main(fpath):
    input_fasta = fpath
    outfile_prefix = os.path.splitext(fpath)[0]
    output_f = outfile_prefix + '_guide_library.csv'

    guide_dic = {} #guide sequence and features
    # handle multiple fasta sequences, support different genes (but don't put different isoform sequences for the same gene!)
    for record in SeqIO.parse(input_fasta, "fasta"):
        sequence = record.seq
        transcript_ID = record.id
        title = str(record.description)

        #design guides
        gene_guide_dic = {} #gene-specific guide dic
        trans_seq = sequence
        for i in range(0,len(trans_seq)-29):
            target = trans_seq[i:i+30]
            spacer = target.reverse_complement()
            # replace U and upper
            spacer = (str(spacer)).upper()
            spacer = spacer.replace("U", "T")


            #flanking seq
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
            # replace U and upper
            nearby_seq_all_15 = (str(nearby_seq_all_15)).upper()
            nearby_seq_all_15 = nearby_seq_all_15.replace("U", "T")
            target = (str(target)).upper()
            target = target.replace("U", "T")

            if str(spacer) not in gene_guide_dic.keys(): #first occur
                gene_guide_dic[str(spacer)]=[str(transcript_ID),str(spacer),[i],1]+[str(target),str(nearby_seq_all_15)]
                                            #transcript_id, spacer,pos_list,target_pos_num,
                                            # target seq, target seq with 15nt flanks 
            else: #guide targeting the same transcript
                gene_guide_dic[str(spacer)][2].append(i)
                gene_guide_dic[str(spacer)][3] += 1
                
        for guide in gene_guide_dic.keys():
            if guide not in guide_dic.keys():
                #target_percent = float(gene_guide_dic[guide][3])/trans_num
                guide_dic[guide]= gene_guide_dic[guide]
            else: #guide target other genes
                guide_dic.pop(guide) #remove the guide with off targets
        
    with open(output_f,'w') as outf:
        writer = csv.writer(outf)
        writer.writerow(['transcript id','guide','target_pos_list','target pos num',
                         'target_seq','nearby_seq_all_15'])
        for g in guide_dic.keys():
            writer.writerow(guide_dic[g])


    # make Linearfold input files
    # guide mfe input
    standard_prefix = "CAAGTAAACCCCTACCAACTGGTCGGGGTTTGAAAC"
    df = pd.read_csv(output_f)
    num_examples = len(df['guide'].values)
    guideseq = df['guide'].values  

    outfile = outfile_prefix + '_linfold_guides_input.fasta'
    with open(outfile, 'w') as outF:
        for i in range(num_examples):
            outF.write(">guide #" + str(i + 1) + ", " +  df.iloc[i, 0] + "\n")
            outF.write(standard_prefix + guideseq[i] + "\n")

    #target with flank 15 nt
    #native
    nearby_seq_all = df['nearby_seq_all_15'].values
    outfile = outfile_prefix + "_linfold_guides_nearby15_input.fasta"
    with open(outfile, 'w') as outF:
        for i in range(num_examples):
            outF.write(">guide #" + str(i + 1) + ", " +  df.iloc[i, 0] + "\n")
            outF.write(nearby_seq_all[i] + "\n")
            
    # with constraints
    guide_rv = df['target_seq'].values
    outfile = outfile_prefix + "_linfold_guides_constraints_nearby15_input.fasta"
    with open(outfile, 'w') as outF:
        for i in range(num_examples):
            outF.write(">guide #" + str(i + 1) + ", " +  df.iloc[i, 0] + "\n")
            outF.write(nearby_seq_all[i] + "\n")
            target_index = nearby_seq_all[i].find(guide_rv[i])
            constraints = "?"*target_index + "."*30 + "?"*(len(nearby_seq_all[i])-30-target_index)
            outF.write(constraints + "\n")


if __name__ == '__main__':
    main(str(sys.argv[1]))
