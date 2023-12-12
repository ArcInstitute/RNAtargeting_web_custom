import os
import uuid
import subprocess
import h5py
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
from scipy.stats import linregress
from scipy import stats
import tensorflow as tf
from tensorflow import keras



base_positions = {
    'A': 0,
    'T': 1,
    'C': 2,
    'G': 3,
    0: 'A',
    1: 'T',
    2: 'C',
    3: 'G',
}

flip_dict = {
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C',
}

linearfold_positions = {
    '.': 0,
    '(': 1,
    ')': 2,
    ' ': -1,
    0: '.',
    1: '(',
    2: ')',
    -1: ' ',
}

RESULTS_DIR = 'results/'

def read_byte_string(fpath, tmpdir_name):
    """
    Reads a byte string and writes it to a temporary file.
    """
    try:
        tmpfile = os.path.join(tmpdir_name, str(uuid.uuid4()) + '.fasta')
        with open(tmpfile, 'w') as tmp:
            tmp.write(fpath.decode('utf-8'))
    except Exception as e:
        raise Exception(f"The input file is not a valid FASTA file: {e}")
    return tmpfile  

def run_subprocess(cmd):
    """
    Runs a subprocess and returns the output and errors.
    """
    try:
        # Start the subprocess
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )

        # Wait for the process to complete and capture output and errors
        stdout, stderr = process.communicate() 

        # Check if the subprocess was successful
        if process.returncode != 0:
            print("Error:", stderr)
        else:
            print("Output:", stdout)
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    return stdout

def encoded_nuc_to_str(encoded_seq):
    indices = np.argmax(encoded_seq, axis=1)
    return ''.join([base_positions[i] for i in indices])

def save_to_results(model_name, dataset_name, regre, kfold, split, figure_name):
    dataset_folder = RESULTS_DIR + dataset_name + '/'
    if regre:
    	model_folder = dataset_folder + model_name + '_regression' + '/'
    else:
    	model_folder = dataset_folder + model_name + '_classification' + '/'
    #model_folder = dataset_folder + model_name + '/'


    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    if kfold == None:
        plt.savefig('%s%s.svg' % (model_folder, figure_name), bbox_inches='tight')
    else:
        #plt.savefig('%s%s_%sof%s.png' % (model_folder, figure_name, split, kfold), dpi=600, bbox_inches='tight')
        plt.savefig('%s%s_%sof%s.svg' % (model_folder, figure_name, split, kfold), format="svg", bbox_inches='tight')


def get_outputs_and_labels(model, dataset, sigmoid=True, invert=False):
    if sigmoid:
        outputs = np.array(list(tf.sigmoid(model.predict(dataset)).numpy().flat))
    else:
        outputs = np.array(model.predict(dataset).flat)
    true_labels = np.array([label for (input, label) in dataset.unbatch().as_numpy_iterator()])

    if invert:
        outputs = 1 - outputs
        true_labels = 1 - true_labels

    return outputs, true_labels



def classification_analysis_new(testpath, model: keras.Model, test_dataset: tf.data.Dataset, regre, kfold, split, guidelength, model_name='', dataset_name=''):
    test_inputs = [inputs for (inputs, label) in test_dataset.unbatch()]
    test_outputs = [label for (inputs, label) in test_dataset.unbatch()]

    
    if len(test_inputs[0]) == 2:
        test_sequences =[np.array(sequences) for (sequences, features) in test_inputs]
        test_features = [features for (sequences, features) in test_inputs]
    else:
        test_sequences = [np.array(sequences[0]) for sequences in test_inputs]
        
    # def encoded_nuc_to_str(encoded_seq):
    #     indices = np.argmax(encoded_seq, axis=1)
    #     return ''.join([base_positions[i] for i in indices])
    
    outputs, labels = get_outputs_and_labels(model, test_dataset, sigmoid=True)
    
    df = pd.read_csv(testpath)

    test_predic =[]

    for i in range(len(outputs)):
        nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
        df1 = df[df['guide'] == nuc_sequence]
        if df1.empty:
            target_gene = ' '
        else:
            target_gene = df1['gene'].values[0]
        test_predic.append([nuc_sequence,target_gene,outputs[i],labels[i]])
    test_df = pd.DataFrame(test_predic, columns = ['spacer sequence', 'target gene','predicted_value_sigmoid','true label'])
    test_df['output rank'] = test_df['predicted_value_sigmoid'].rank(ascending=False)    

    
    dataset_folder = RESULTS_DIR + dataset_name + '/'
    #model_folder = dataset_folder + model_name + '/'
    model_folder = dataset_folder + model_name + '_classification' + '/'
    
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
        
    if kfold == None:
        test_df.to_csv('%s%s.csv' % (model_folder, "test_prediction"))
    else:
        test_df.to_csv('%s%s_%s_%sof%s.csv' % (model_folder, "test_prediction", 'gl-'+str(guidelength), split, kfold))


def classification_analysis(model, test_dataset, model_name, dataset_name, regre, kfold, split, guidelength):
    test_inputs = [inputs for (inputs, label) in test_dataset.unbatch()]
    test_outputs = [label for (inputs, label) in test_dataset.unbatch()]

    
    if len(test_inputs[0]) == 2:
        test_sequences =[np.array(sequences) for (sequences, features) in test_inputs]
        test_features = [features for (sequences, features) in test_inputs]
    else:
        test_sequences = [np.array(sequences[0]) for sequences in test_inputs]
        
    # def encoded_nuc_to_str(encoded_seq):
    #     indices = np.argmax(encoded_seq, axis=1)
    #     return ''.join([base_positions[i] for i in indices])
    
    outputs, labels = get_outputs_and_labels(model, test_dataset, sigmoid=True)
    
    #dataset_filtered_csv_path = 'dataset/integrated_guide_feature_np_vivo_all.csv' 
    #dataset_filtered_csv_path = 'dataset/integrated_guide_feature_filtered_new_ver3.csv'
    dataset_filtered_csv_path = 'dataset/integrated_guide_feature_filtered_f24_mismatch3_iso_percent_cds_floats_rnafe.csv'
    df = pd.read_csv(dataset_filtered_csv_path)
    df['raw_ratio_pct_rank'] = df['raw ratio'].rank(pct=True)

    test_predic =[]
    if int(guidelength)<30: #chop guide
        for i in range(len(outputs)):
            nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
            test_predic.append([nuc_sequence,outputs[i],labels[i]])
        test_df = pd.DataFrame(test_predic, columns = ['spacer sequence chopped', 'predicted_value_sigmoid','true ratio binary'])
        test_df['output rank'] = test_df['predicted_value_sigmoid'].rank(ascending=False)    
    else:
        for i in range(len(outputs)):
            nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
            df1 = df[df['guide'] == nuc_sequence]
            rank_gene = df1['relative_ratio'].values[0]
            #rank_gene = df.loc[df['guide'] == nuc_sequence]['relative_ratio']
            rank_ratio = df1['raw_ratio_pct_rank'].values[0]
            target_gene = df1['gene'].values[0]
            #info = info + list(test_features[i]) #single value inputs
            #info = [nuc_sequence,outputs[i],labels[i],rank_ratio,rank_gene]
            test_predic.append([nuc_sequence,target_gene,outputs[i],labels[i],rank_ratio,rank_gene])    
        #single_value_inputs = ['is_5UTR', 'is_CDS', 'is_3UTR', 'RNAseq', 'linearfold value']
        test_df = pd.DataFrame(test_predic, columns = ['spacer sequence', 'gene','predicted_value_sigmoid','true ratio binary','rank_by_raw_ratio','rank by gene'])
        test_df['output rank'] = test_df['predicted_value_sigmoid'].rank(ascending=False)
    
    dataset_folder = RESULTS_DIR + dataset_name + '/'
    #model_folder = dataset_folder + model_name + '/'
    model_folder = dataset_folder + model_name + '_classification' + '/'
    
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
        
    if kfold == None:
        test_df.to_csv('%s%s.csv' % (model_folder, "test_prediction"))
    else:
        test_df.to_csv('%s%s_%s_%sof%s.csv' % (model_folder, "test_prediction", 'gl-'+str(guidelength), split, kfold))

        #false positive plots
    if int(guidelength) == 30:
        thres_list = [0.5, 0.8, 0.9]
        print('thres_stats')
        for thres in thres_list:
            print(thres)
            df_pre_good = test_df[test_df['predicted_value_sigmoid']>thres]
            real_raw_ratio_rank = df_pre_good['rank_by_raw_ratio'].values
            guide_num = len(real_raw_ratio_rank)
            print('guide_num '+str(guide_num))
            if guide_num>3:
                real_raw_ratio_pct = np.percentile(real_raw_ratio_rank, [25,50,75])
                print('raw_ratio_rank_pct: 25%, 50%, 75%')
                print(real_raw_ratio_pct)
                real_gene_rank = df_pre_good['rank by gene'].values
                real_gener_pct = np.percentile(real_gene_rank, [25,50,75])
                print('gene_rank_pct: 25%, 50%, 75%')
                print(real_gener_pct)
            true_bin_ratio = df_pre_good['true ratio binary'].values
            num_real_gg = np.count_nonzero(true_bin_ratio)
            if len(true_bin_ratio)>0:
                gg_ratio = num_real_gg/len(true_bin_ratio)
                print('true good guide percent '+str(gg_ratio))

            #plot predicted positive distribution, 20% cutoff
            tp_df = df_pre_good[df_pre_good['true ratio binary']==1]
            fp_df = df_pre_good[df_pre_good['true ratio binary']==0]

            #fp_relative_ratio = (fp_df['rank_by_raw_ratio']*100).to_numpy()
            #tp_relative_ratio = (tp_df['rank_by_raw_ratio']*100).to_numpy()
            fp_relative_ratio = (fp_df['rank by gene']*100).to_numpy()
            tp_relative_ratio = (tp_df['rank by gene']*100).to_numpy()
        
            plt.clf()
            plt.hist(fp_relative_ratio, bins = range(0, 101, 1), label='false positive distribution')
            plt.hist(tp_relative_ratio, bins = range(0, 101, 1), label='true positive distribution')
            plt.xlim(0,100)
            plt.xlabel('guide percentile rank')
            plt.ylabel('count')
            plt.text(1,1,'threshold'+str(thres)+' true positive ratio: '+str(round(gg_ratio, 2)),fontsize=10)
            plt.title("Distribution of predicted good guides' rank ")        
            plt.legend(loc='upper right')
            save_to_results(model_name, dataset_name, regre, kfold, split, 'predicted_positives_ratio_distribution_thres'+str(thres))
    
        #top 10 predicted plots    
        test_df_top10 = test_df[test_df['output rank']<=10]
        test_df_rank = test_df_top10['output rank'].values
        test_df_raw_r =test_df_top10['rank_by_raw_ratio'].values
        test_df_gene_r = test_df_top10['rank by gene'].values
        plt.clf()
        #plt.plot(test_df_rank, test_df_raw_r,'o', color='black',label='raw ratio percentile')
        #plt.plot(test_df_rank, test_df_gene_r,'o', color='red',label='gene rank percentile')
        plt.plot(test_df_rank, test_df_gene_r,'o')
        plt.ylim(0,1)
        plt.legend(loc='upper right')
        plt.title("Top 10 predicted good guides' true rank")
        plt.xticks(ticks=np.arange(11),labels=np.arange(11))
        plt.xlabel('predicted rank')
        plt.ylabel('true percentile rank')
        save_to_results(model_name, dataset_name, regre, kfold, split, 'top10_predict')
       
        #previous comfusion matrix (thres = 0.5)    
        # new_array = np.zeros((len(test_sequences), test_sequences[0].shape[0], test_sequences[0].shape[1]))
        # np.copyto(new_array, test_sequences)
        # test_sequences = new_array

        # outputs, labels = get_outputs_and_labels(model, test_dataset, sigmoid=True)
        # rounded_outputs = np.rint(outputs)
        # cm = confusion_matrix(labels, np.rint(outputs))
        # true_positives = np.where(np.logical_and(rounded_outputs==1, labels == 1))[0]
        # true_negatives = np.where(np.logical_and(rounded_outputs==0, labels == 0))[0]
        # false_positives = np.where(np.logical_and(rounded_outputs==1, labels == 0))[0]
        # false_negatives = np.where(np.logical_and(rounded_outputs==0, labels == 1))[0]

        # true_positive_sequences = test_sequences[true_positives]
        # true_negative_sequences = test_sequences[true_negatives]
        # false_positive_sequences = test_sequences[false_positives]
        # false_negative_sequences = test_sequences[false_negatives]

        # tp = [encoded_nuc_to_str(encoded) for encoded in true_positive_sequences]
        # tn= [encoded_nuc_to_str(encoded) for encoded in true_negative_sequences]
        # fp = [encoded_nuc_to_str(encoded) for encoded in false_positive_sequences]
        # fn = [encoded_nuc_to_str(encoded) for encoded in false_negative_sequences]

        # real_positives = tp + fn

        # fp_df = df[df['guide'].isin(fp)]
        # rp_df = df[df['guide'].isin(real_positives)]
        # tp_df = df[df['guide'].isin(tp)]

        # print('fp num:'+str(len(fp_df['guide'].values)))
        # print('tp num:'+str(len(tp_df['guide'].values)))
        # fp_ratio = len(fp_df['guide'].values)/(len(fp_df['guide'].values)+len(tp_df['guide'].values))
        # print('fp_ratio:'+str(fp_ratio))

        #summary_stats = fp_df.describe()
        #print("false positives")
        #print(summary_stats)
        #print(summary_stats['relative_ratio'])

        #summary_stats = rp_df.describe()
        #print("actual positives")
        #print(summary_stats)
        #print(summary_stats['relative_ratio'])

        

def classification_analysis_cd(model, test_dataset, model_name, dataset_name, regre, kfold, split, guidelength):
    test_inputs = [inputs for (inputs, label) in test_dataset.unbatch()]
    test_outputs = [label for (inputs, label) in test_dataset.unbatch()]

    
    if len(test_inputs[0]) == 2:
        test_sequences =[np.array(sequences) for (sequences, features) in test_inputs]
        test_features = [features for (sequences, features) in test_inputs]
    else:
        test_sequences = [np.array(sequences[0]) for sequences in test_inputs]
        
    # def encoded_nuc_to_str(encoded_seq):
    #     indices = np.argmax(encoded_seq, axis=1)
    #     return ''.join([base_positions[i] for i in indices])
    
    outputs, labels = get_outputs_and_labels(model, test_dataset, sigmoid=True)
    
    #dataset_filtered_csv_path = 'dataset/cdscreen_filtered_features_ratios.csv'
    dataset_filtered_csv_path = 'dataset/cdscreen_filtered_t1_new_features_ratios.csv'
    df = pd.read_csv(dataset_filtered_csv_path)
    #df['raw_ratio_pct_rank'] = df['raw ratio'].rank(pct=True)
    
    test_predic =[]
    for i in range(len(outputs)):
        info =[]
        nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
        info.append(nuc_sequence)
        df1 = df[df['guide'] == nuc_sequence]
        #rank_gene = df1['relative_ratio'].values[0]
        #rank_ratio_nor3 = df1['bin1_to_sum_bin14_rank'].values[0]
        rank_ratio_r3 = df1['t1_bin1_to_sum_bin14_rank_withr3'].values[0]
        #classi_nor3 = df1['binary_relative_ratio_norep3'].values[0]
        classi_r3 = df1['t1_binary_relative_ratio_withrep3'].values[0]

        info = info + [outputs[i],labels[i],rank_ratio_r3,classi_r3]
        test_predic.append(info)    
    
    #single_value_inputs = ['is_5UTR', 'is_CDS', 'is_3UTR', 'RNAseq', 'linearfold value']
    test_df = pd.DataFrame(test_predic, columns = ['spacer sequence', 'predicted_value_sigmoid','true ratio binary','rank_ratio_r3','binary_ratio_r3'])
    test_df['output rank'] = test_df['predicted_value_sigmoid'].rank(ascending=False)
    
    dataset_folder = RESULTS_DIR + dataset_name + '/'
    #model_folder = dataset_folder + model_name + '/'
    model_folder = dataset_folder + model_name + '_classification' + '/'
    
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    if kfold == None:
        test_df.to_csv('%s%s.csv' % (model_folder, "test_prediction"))
    else:
        test_df.to_csv('%s%s_%sof%s.csv' % (model_folder, "test_prediction", split, kfold))        

    # thres_list = [0.8, 0.9,0.95]
    # print('thres_stats')
    # for thres in thres_list:
    #     print(thres)
    #     df_pre_good = test_df[test_df['predicted_value_sigmoid']>thres]
    #     real_raw_ratio_rank = df_pre_good['rank_ratio_r3'].values
    #     #real_raw_ratio_rank_nor3 = df_pre_good['rank_ratio_nor3'].values
    #     guide_num = len(real_raw_ratio_rank)
    #     print('guide_num '+str(guide_num))
    #     if guide_num>3:
    #         real_raw_ratio_pct = np.percentile(real_raw_ratio_rank, [25,50,75])
    #         print('ratio_rank_pct_withr3: 25%, 50%, 75%')
    #         print(real_raw_ratio_pct)
    #         #real_raw_ratio_pct_nor3 = np.percentile(real_raw_ratio_rank_nor3, [25,50,75])
    #         #print('ratio_rank_pct_nor3: 25%, 50%, 75%')
    #         #print(real_raw_ratio_pct_nor3)            

    #     true_bin_ratio = df_pre_good['true ratio binary'].values
    #     num_real_gg = np.count_nonzero(true_bin_ratio)
    #     if len(true_bin_ratio)>0:
    #         gg_ratio = num_real_gg/len(true_bin_ratio)
    #         print('true good guide percent '+str(gg_ratio))   
        
    #test_df_top10 = test_df[test_df['output rank']<=10]
    #test_df_rank = test_df_top10['output rank'].values
    #test_df_r_r3 =test_df_top10['rank_ratio_r3'].values
    #test_df_r_nor3 = test_df_top10['rank_ratio_nor3'].values
    #plt.clf()
    #plt.plot(test_df_rank, test_df_r_r3,'o', label='raw ratio percentile')
    #plt.plot(test_df_rank, test_df_r_nor3,'o', color='red',label='raw ratio percentile_without rep3')
    #plt.legend(loc='upper right')
    #plt.xlabel('top 10 predicted guide rank')
    #plt.ylabel('true rank percentile')
    #save_to_results(model_name, dataset_name, regre, kfold, split, 'top10_predict')
           
    #sr, srp = stats.spearmanr(test_df['predicted_value_sigmoid'], (1- test_df['rank_ratio_r3']))
    #print('spearmanr '+str(sr))

def classification_analysis_nbt(model, test_dataset, model_name, dataset_name, regre, kfold, split, guidelength):
    test_inputs = [inputs for (inputs, label) in test_dataset.unbatch()]
    test_outputs = [label for (inputs, label) in test_dataset.unbatch()]

    
    if len(test_inputs[0]) == 2:
        test_sequences =[np.array(sequences) for (sequences, features) in test_inputs]
        test_features = [features for (sequences, features) in test_inputs]
    else:
        test_sequences = [np.array(sequences[0]) for sequences in test_inputs]
        
    # def encoded_nuc_to_str(encoded_seq):
    #     indices = np.argmax(encoded_seq, axis=1)
    #     return ''.join([base_positions[i] for i in indices])
    
    outputs, labels = get_outputs_and_labels(model, test_dataset, sigmoid=True)
    
    dataset_filtered_csv_path = 'dataset/cas13d_nbt-data_filtered_iscds_linf_contra_iso_percent.csv'
    df = pd.read_csv(dataset_filtered_csv_path)
    #df['raw_ratio_pct_rank'] = df['raw ratio'].rank(pct=True)
    
    test_predic =[]
    for i in range(len(outputs)):
        info =[]
        nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
        info.append(nuc_sequence)
        df1 = df[df['guide'] == nuc_sequence]
        rank_ratio = df1['relative_ratio'].values[0]
        classi = df1['binary_relative_ratio'].values[0]

        info = info + [outputs[i],labels[i],rank_ratio,classi]
        test_predic.append(info)    
    
    #single_value_inputs = ['is_5UTR', 'is_CDS', 'is_3UTR', 'RNAseq', 'linearfold value']
    test_df = pd.DataFrame(test_predic, columns = ['spacer sequence', 'predicted_value_sigmoid','true ratio binary','rank_ratio','binary_ratio_badg'])
    test_df['output rank'] = test_df['predicted_value_sigmoid'].rank(ascending=False)
    
    dataset_folder = RESULTS_DIR + dataset_name + '/'
    #model_folder = dataset_folder + model_name + '/'
    model_folder = dataset_folder + model_name + '_classification' + '/'
    
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    if kfold == None:
        test_df.to_csv('%s%s.csv' % (model_folder, "test_prediction"))
    else:
        test_df.to_csv('%s%s_%sof%s.csv' % (model_folder, "test_prediction", split, kfold))

    # thres_list = [0.8, 0.9,0.95]
    # print('thres_stats')
    # for thres in thres_list:
    #     print(thres)
    #     df_pre_good = test_df[test_df['predicted_value_sigmoid']>thres]
    #     real_ratio_rank = df_pre_good['rank_ratio'].values
    #     guide_num = len(real_ratio_rank)
    #     print('guide_num '+str(guide_num))
    #     if guide_num>3:
    #         real_raw_ratio_pct = np.percentile(real_ratio_rank, [25,50,75])
    #         print('ratio_rank_pct: 25%, 50%, 75%')
    #         print(real_raw_ratio_pct)
          
    #     true_bin_ratio = df_pre_good['true ratio binary'].values
    #     num_real_gg = np.count_nonzero(true_bin_ratio)
    #     if len(true_bin_ratio)>0:
    #         gg_ratio = num_real_gg/len(true_bin_ratio)
    #         print('true good guide percent '+str(gg_ratio))   

        
    # test_df_top10 = test_df[test_df['output rank']<=10]
    # test_df_rank = test_df_top10['output rank'].values
    # test_df_ratio =test_df_top10['rank_ratio'].values

    # plt.plot(test_df_rank, test_df_ratio,'o', color='black',label='gene ratio percentile')
    # #plt.plot(test_df_rank, test_df_r_nor3,'o', color='red',label='raw ratio percentile_without rep3')
    # plt.legend(loc='upper right')
    # plt.xlabel('top 10 predicted guide rank')
    # plt.ylabel('true rank percentile')
    # save_to_results(model_name, dataset_name, regre, kfold, split, 'top10_predict')

    sr, srp = stats.spearmanr(test_df['predicted_value_sigmoid'], (1- test_df['rank_ratio']))
    print('spearmanr '+str(sr))
    #print('spearmanr_p '+str(srp))
   

def get_classification_metrics(model: keras.Model, test_set: tf.data.Dataset, fig, ax1, ax2, regre, kfold, split, guidelength, model_name='', dataset_name='', 
                               save=True):


    outputs, labels = get_outputs_and_labels(model, test_set, sigmoid=True)
    fig.suptitle('AUC and PRC')
    score = roc_auc_score(labels, outputs)
    fpr, tpr, _ = roc_curve(labels, outputs)
    print('AUROC '+str(score))
    average_precision = average_precision_score(labels, outputs)
    precision, recall, thres_prc  = precision_recall_curve(labels, outputs)
    print('AUPRC '+str(average_precision))
    sr, srp = stats.spearmanr(outputs, labels)
    print('spearmanr '+str(sr))
    #thres_list = [0.5, 0.8, 0.9,0.95]
    #print('thres_prc_stats')
    #for thres in thres_list:
    #    print(thres)
    #    for i in range(len(thres_prc)):
    #        if thres_prc[i] == thres:
    #            print('precision '+str(precision[i]))

    ax1.plot(fpr, tpr, label='%s, AUROC=%f' % (model_name.split(':')[1], score))
    ax2.plot(precision, recall, label='%s, AUPRC=%f' % (model_name.split(':')[1], average_precision))
    if save:
        model_name = model_name.split(':')[0]
        ax1.legend(bbox_to_anchor=(0, 1.5), loc = 'upper left', fontsize=6)
        ax1.set_title('Test Set AUROC')
        ax1.set_xlabel('false positive rate')
        ax1.set_ylabel('true positive rate')
        ax1.set_aspect('equal')

        ax2.legend(bbox_to_anchor=(0, 1.5), loc = 'upper left', fontsize=6)
        ax2.set_title('Test Set AUPRC')
        ax2.set_xlabel('precision')
        ax2.set_ylabel('recall')
        ax2.set_aspect('equal')
        plt.tight_layout()

        save_to_results(model_name, dataset_name, regre, kfold, split, 'roc')

        plt.clf()
        cm = confusion_matrix(labels, np.rint(outputs))
        df_cm = pd.DataFrame(cm, range(2), range(2))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, fmt='d')
        plt.xlabel("predicted labels")
        plt.ylabel("true labels")
        #save_to_results(model_name, dataset_name, regre, kfold, split, 'confusion_matrix')

        plt.clf()
        classification_analysis(model, test_set, model_name, dataset_name, regre, kfold, split, guidelength)

def get_classification_metrics_cd(model: keras.Model, test_set: tf.data.Dataset, fig, ax1, ax2, regre, kfold, split, guidelength, model_name='', dataset_name='', 
                               save=True):


    outputs, labels = get_outputs_and_labels(model, test_set, sigmoid=True)
    fig.suptitle('AUC and PRC')
    score = roc_auc_score(labels, outputs)
    fpr, tpr, _ = roc_curve(labels, outputs)
    print('AUROC '+str(score))
    average_precision = average_precision_score(labels, outputs)
    precision, recall, thres_prc  = precision_recall_curve(labels, outputs)
    print('AUPRC '+str(average_precision))


    ax1.plot(fpr, tpr, label='model %s, AUROC=%f' % (model_name, score), )
    ax2.plot(precision, recall, label='model %s, AUPRC=%f' % (model_name, average_precision))
    if save:
        ax1.legend(bbox_to_anchor=(0, 1.5), loc = 'upper left', fontsize=6)
        ax1.set_title('Test Set AUROC')
        ax1.set_xlabel('false positive rate')
        ax1.set_ylabel('true positive rate')
        ax1.set_aspect('equal')

        ax2.legend(bbox_to_anchor=(0, 1.5), loc = 'upper left', fontsize=6)
        ax2.set_title('Test Set AUPRC')
        ax2.set_xlabel('precision')
        ax2.set_ylabel('recall')
        ax2.set_aspect('equal')
        plt.tight_layout()

        save_to_results(model_name, dataset_name, regre, kfold, split, 'roc')

        #plt.clf()
        #cm = confusion_matrix(labels, np.rint(outputs))
        #df_cm = pd.DataFrame(cm, range(2), range(2))
        #sn.set(font_scale=1.4)
        #sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, fmt='d')
        #plt.xlabel("predicted labels")
        #plt.ylabel("true labels")
        #save_to_results(model_name, dataset_name, regre, kfold, split, 'confusion_matrix')

        plt.clf()
        classification_analysis_cd(model, test_set, model_name, dataset_name, regre, kfold, split, guidelength)


def get_classification_metrics_nbt(testpath, model: keras.Model, test_set: tf.data.Dataset, fig, ax1, ax2, regre, kfold, split, guidelength, model_name='', dataset_name='', 
                               save=True):


    outputs, labels = get_outputs_and_labels(model, test_set, sigmoid=True)
    fig.suptitle('AUC and PRC')
    score = roc_auc_score(labels, outputs)
    fpr, tpr, _ = roc_curve(labels, outputs)
    print('AUROC '+str(score))
    average_precision = average_precision_score(labels, outputs)
    precision, recall, thres_prc  = precision_recall_curve(labels, outputs)
    print('AUPRC '+str(average_precision))


    ax1.plot(fpr, tpr, label='model %s, AUROC=%f' % (model_name, score), )
    ax2.plot(precision, recall, label='model %s, AUPRC=%f' % (model_name, average_precision))
    if save:
        ax1.legend(bbox_to_anchor=(0, 1.5), loc = 'upper left', fontsize=6)
        ax1.set_title('Test Set AUROC')
        ax1.set_xlabel('false positive rate')
        ax1.set_ylabel('true positive rate')
        ax1.set_aspect('equal')

        ax2.legend(bbox_to_anchor=(0, 1.5), loc = 'upper left', fontsize=6)
        ax2.set_title('Test Set AUPRC')
        ax2.set_xlabel('precision')
        ax2.set_ylabel('recall')
        ax2.set_aspect('equal')
        plt.tight_layout()

        save_to_results(model_name, dataset_name, regre, kfold, split, 'roc')

        plt.clf()
        classification_analysis_new(testpath, model, test_set, regre, kfold, split, guidelength, model_name, dataset_name)

        

def get_regression_metrics(model: keras.Model, test_set: tf.data.Dataset, regre, kfold, split, model_name='', dataset_name='', save=True):
    outputs, labels = get_outputs_and_labels(model, test_set, sigmoid=False)
    test_inputs = [inputs for (inputs, label) in test_set.unbatch()]
    test_outputs = [label for (inputs, label) in test_set.unbatch()]
    if len(test_inputs[0]) == 2:
        test_sequences =[np.array(sequences) for (sequences, features) in test_inputs]
        test_features = [features for (sequences, features) in test_inputs]
    else:
        test_sequences = [np.array(sequences[0]) for sequences in test_inputs]
        
    # def encoded_nuc_to_str(encoded_seq):
    #     indices = np.argmax(encoded_seq, axis=1)
    #     return ''.join([base_positions[i] for i in indices])
    
    #dataset_filtered_csv_path = 'dataset/integrated_guide_feature_filtered_new_ver3.csv'
    dataset_filtered_csv_path = 'dataset/integrated_guide_feature_filtered_f24_mismatch3_rnafe.csv'
    df = pd.read_csv(dataset_filtered_csv_path)
    #df['raw_ratio_pct_rank'] = df['ratio'].rank(pct=True)
    df['raw_ratio_pct_rank'] = df['raw ratio'].rank(pct=True)
    
    test_predic =[]
    for i in range(len(outputs)):
        info =[]
        nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
        info.append(nuc_sequence)
        df1 = df[df['guide'] == nuc_sequence]
        rank_gene = df1['relative_ratio'].values[0]
        rank_ratio = df1['raw_ratio_pct_rank'].values[0]
        #info = info + list(test_features[i]) #single value inputs
        info = info + [outputs[i],labels[i],rank_ratio,rank_gene]
        test_predic.append(info)    
    
    #single_value_inputs = ['is_5UTR', 'is_CDS', 'is_3UTR', 'RNAseq', 'linearfold value']
    test_df = pd.DataFrame(test_predic, columns = ['spacer sequence', 'predicted_value','true relative ratio','rank_by_raw_ratio','rank by gene'])
    test_df['output rank'] = test_df['predicted_value'].rank()
    
    dataset_folder = RESULTS_DIR + dataset_name + '/'
    #model_folder = dataset_folder + model_name + '/'
    model_folder = dataset_folder + model_name + '_regression' + '/'
    
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
        
    #if kfold == None:
    #    test_df.to_csv('%s%s.csv' % (model_folder, "test_prediction"))
    #else:
    #    test_df.to_csv('%s%s_%sof%s.csv' % (model_folder, "test_prediction", split, kfold))

    thres_list = [0.8, 0.9,0.95]
    print('thres_stats')
    for thres in thres_list:
        print(thres)
        df_pre_good = test_df[test_df['predicted_value']<(1-thres)]
        real_raw_ratio_rank = df_pre_good['true relative ratio'].values
        guide_num = len(real_raw_ratio_rank)
        print('guide_num '+str(guide_num))
        if guide_num>3:
            real_raw_ratio_pct = np.percentile(real_raw_ratio_rank, [25,50,75])
            print('raw_ratio_rank_pct: 25%, 50%, 75%')
            print(real_raw_ratio_pct)
        true_goodg = df_pre_good[df_pre_good['true relative ratio']<0.2]
        num_real_gg = len(true_goodg['true relative ratio'].values)
        if guide_num>0:
            gg_ratio = num_real_gg/guide_num
            print('true good guide percent '+str(gg_ratio))   


    test_df_top10 = test_df[test_df['output rank']<=10]
    test_df_rank = test_df_top10['output rank'].values
    test_df_raw_r =test_df_top10['rank_by_raw_ratio'].values
    test_df_gene_r = test_df_top10['rank by gene'].values
    
    plt.plot(test_df_rank, test_df_raw_r,'o', color='black',label='raw ratio percentile')
    plt.plot(test_df_rank, test_df_gene_r,'o', color='red',label='gene rank percentile')
    plt.legend(loc='upper right')
    plt.xlabel('top 10 predicted guide rank')
    plt.ylabel('true rank percentile')
    save_to_results(model_name, dataset_name, regre, kfold, split, 'top10_predict')
    
    
   
    _, _, r_value, _, _ = linregress(labels, outputs)

    mean_error = float(np.mean(np.abs(np.array(outputs) - np.array(labels))))
    msev = float(np.mean((np.array(outputs) - np.array(labels))**2))
    
    plt.clf()
    plt.plot(labels, labels)
    plt.plot(labels, outputs, 'r.', label='mean_error=%.4f, MSE=%.4f,r^2=%.4f' % (mean_error, msev,r_value ** 2))
    print('MSE '+str(msev))
    print('R squared '+ str(r_value ** 2))

    if save:
        plt.legend()
        plt.xlabel('relative ratio (scaled)')
        plt.ylabel('predicted ratio')
        save_to_results(model_name, dataset_name, regre, kfold, split, 'test-prediction-scatter')

        plt.clf()
        outputs, labels = get_outputs_and_labels(model, test_set, sigmoid=True)
        outputs = np.rint(outputs)
        labels = np.rint(labels)
        cm = confusion_matrix(labels, outputs)
        df_cm = pd.DataFrame(cm, range(2), range(2))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, fmt='d')
        plt.xlabel("predicted labels")
        plt.ylabel("true labels")
        #save_to_results(model_name, dataset_name, regre, kfold, split, 'binarized_confusion_matrix')

        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('AUC and PRC')
        score = roc_auc_score(labels, outputs)
        fpr, tpr, _ = roc_curve(labels, outputs)
        average_precision = average_precision_score(labels, outputs)
        precision, recall, _  = precision_recall_curve(labels, outputs)
        ax1.plot(fpr, tpr, label='model %s, AUROC=%f' % (model_name, score), )
        ax2.plot(precision, recall, label='model %s, AUPRC=%f' % (model_name, average_precision))

        ax1.legend(loc = 'upper left', fontsize=3)
        ax1.set_title('Test Set AUROC')
        ax1.set_xlabel('false positive rate')
        ax1.set_ylabel('true positive rate')

        ax2.legend(loc = 'upper left', fontsize=3)
        ax2.set_title('Test Set AUPRC')
        ax2.set_xlabel('precision')
        ax2.set_ylabel('recall')
        #save_to_results(model_name, dataset_name, regre, kfold, split, 'binarized_roc')


def get_regression_metrics_nbt(model: keras.Model, test_set: tf.data.Dataset, regre, kfold, split, model_name='', dataset_name='', save=True):
    outputs, labels = get_outputs_and_labels(model, test_set, sigmoid=False)
    test_inputs = [inputs for (inputs, label) in test_set.unbatch()]
    test_outputs = [label for (inputs, label) in test_set.unbatch()]
    if len(test_inputs[0]) == 2:
        test_sequences =[np.array(sequences) for (sequences, features) in test_inputs]
        test_features = [features for (sequences, features) in test_inputs]
    else:
        test_sequences = [np.array(sequences[0]) for sequences in test_inputs]
        
    # def encoded_nuc_to_str(encoded_seq):
    #     indices = np.argmax(encoded_seq, axis=1)
    #     return ''.join([base_positions[i] for i in indices])
    
    dataset_filtered_csv_path = 'dataset/cas13d_nbt-data_combined_input.csv'
    df = pd.read_csv(dataset_filtered_csv_path)
    df["rank_pct"] = df.groupby("Screen")["normCS"].rank(ascending=False,pct=True)
    
    test_predic =[]
    for i in range(len(outputs)):
        info =[]
        nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
        info.append(nuc_sequence)
        df1 = df[df['guideseq'] == nuc_sequence]
        rank_gene = df1['rank_pct'].values[0]
        #info = info + list(test_features[i]) #single value inputs
        info = info + [outputs[i],labels[i],rank_gene]
        test_predic.append(info)    
    
    #single_value_inputs = ['is_5UTR', 'is_CDS', 'is_3UTR', 'RNAseq', 'linearfold value']
    test_df = pd.DataFrame(test_predic, columns = ['spacer sequence', 'predicted_value','true relative ratio','rank by gene'])
    test_df['output rank'] = test_df['predicted_value'].rank()
    
    dataset_folder = RESULTS_DIR + dataset_name + '/'
    #model_folder = dataset_folder + model_name + '/'
    model_folder = dataset_folder + model_name + '_regression' + '/'
    
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
        
    if kfold == None:
        test_df.to_csv('%s%s.csv' % (model_folder, "test_prediction"))
    else:
        test_df.to_csv('%s%s_%sof%s.csv' % (model_folder, "test_prediction", split, kfold))

    test_df_top10 = test_df[test_df['output rank']<=10]
    test_df_rank = test_df_top10['output rank'].values
    test_df_gene_r = test_df_top10['rank by gene'].values
    
    #plt.plot(test_df_rank, test_df_raw_r,'o', color='black',label='raw ratio percentile')
    plt.plot(test_df_rank, test_df_gene_r,'o', color='red',label='gene rank percentile')
    plt.legend(loc='upper right')
    plt.xlabel('top 10 predicted guide rank')
    plt.ylabel('true rank percentile')
    save_to_results(model_name, dataset_name, regre, kfold, split, 'top10_predict')
    
    
   
    _, _, r_value, _, _ = linregress(labels, outputs)

    mean_error = float(np.mean(np.abs(np.array(outputs) - np.array(labels))))
    msev = float(np.mean((np.array(outputs) - np.array(labels))**2))
    sr, srp = stats.spearmanr(np.array(outputs),np.array(labels))
    
    plt.clf()
    plt.plot(labels, labels)
    plt.plot(labels, outputs, 'r.', label='mean_error=%.4f, MSE=%.4f,r^2=%.4f,spearmanr=%.4f' % (mean_error, msev,r_value ** 2,sr))
    print('MSE '+str(msev))
    print('R squared '+ str(r_value ** 2))
    print('spearmanr '+str(sr))
    print('spearmanr_p '+str(srp))

    if save:
        plt.legend()
        plt.xlabel('relative ratio (scaled)')
        plt.ylabel('predicted ratio')
        save_to_results(model_name, dataset_name, regre, kfold, split, 'test-prediction-scatter')


def get_regression_metrics_cd(model: keras.Model, test_set: tf.data.Dataset, regre, kfold, split, model_name='', dataset_name='', save=True):
    outputs, labels = get_outputs_and_labels(model, test_set, sigmoid=False)
    test_inputs = [inputs for (inputs, label) in test_set.unbatch()]
    test_outputs = [label for (inputs, label) in test_set.unbatch()]
    if len(test_inputs[0]) == 2:
        test_sequences =[np.array(sequences) for (sequences, features) in test_inputs]
        test_features = [features for (sequences, features) in test_inputs]
    else:
        test_sequences = [np.array(sequences[0]) for sequences in test_inputs]
        
    # def encoded_nuc_to_str(encoded_seq):
    #     indices = np.argmax(encoded_seq, axis=1)
    #     return ''.join([base_positions[i] for i in indices])
    
    dataset_filtered_csv_path = 'dataset/cdscreen_r1r2_rel_new_targeting_guides.csv'
    df = pd.read_csv(dataset_filtered_csv_path)
    #df["rank_pct"] = df.groupby("gene symbol")["normCS"].rank(ascending=False,pct=True)
    
    test_predic =[]
    for i in range(len(outputs)):
        info =[]
        nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
        info.append(nuc_sequence)
        df1 = df[df['spacer'] == nuc_sequence]
        gene_target = df1['gene symbol'].values[0]
        rank_gene = df1['r2_d10_bin1_to_sum_bin14_rank'].values[0]
        #info = info + list(test_features[i]) #single value inputs
        info = info + [gene_target,outputs[i],labels[i],rank_gene]
        test_predic.append(info)    
    
    #single_value_inputs = ['is_5UTR', 'is_CDS', 'is_3UTR', 'RNAseq', 'linearfold value']
    test_df = pd.DataFrame(test_predic, columns = ['spacer sequence', 'gene symbol','predicted_value','bin1_to_sum_bin14_avg_ratio_rank','r2_d10_bin1_to_sum_bin14_rank'])
    test_df['output rank'] = test_df['predicted_value'].rank()
    
    dataset_folder = RESULTS_DIR + dataset_name + '/'
    #model_folder = dataset_folder + model_name + '/'
    model_folder = dataset_folder + model_name + '_regression' + '/'
    
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
        
    if kfold == None:
        test_df.to_csv('%s%s.csv' % (model_folder, "test_prediction"))
    else:
        test_df.to_csv('%s%s_%sof%s.csv' % (model_folder, "test_prediction", split, kfold))

    test_df_top10 = test_df[test_df['output rank']<=10]
    test_df_rank = test_df_top10['output rank'].values
    test_df_gene_r = test_df_top10['bin1_to_sum_bin14_avg_ratio_rank'].values
    
    #plt.plot(test_df_rank, test_df_raw_r,'o', color='black',label='raw ratio percentile')
    plt.plot(test_df_rank, test_df_gene_r,'o', color='red',label='gene rank percentile')
    plt.legend(loc='upper right')
    plt.xlabel('top 10 predicted guide rank')
    plt.ylabel('true rank percentile: bin1_to_sum_bin14_avg_ratio_rank')
    save_to_results(model_name, dataset_name, regre, kfold, split, 'top10_predict')
    
    
   
    _, _, r_value, _, _ = linregress(labels, outputs)

    mean_error = float(np.mean(np.abs(np.array(outputs) - np.array(labels))))
    msev = float(np.mean((np.array(outputs) - np.array(labels))**2))
    cd58_df = test_df[test_df['gene symbol']=='CD58']
    cd58_sr,srp1 = stats.spearmanr(cd58_df['predicted_value'].values, cd58_df['bin1_to_sum_bin14_avg_ratio_rank'].values)
    cd81_df = test_df[test_df['gene symbol']=='CD81']
    cd81_sr,srp2 = stats.spearmanr(cd81_df['predicted_value'].values, cd81_df['bin1_to_sum_bin14_avg_ratio_rank'].values)
    #sr, srp = stats.spearmanr(np.array(outputs),np.array(labels))
    
    plt.clf()
    plt.plot(labels, labels)
    plt.plot(labels, outputs, 'r.', label='mean_error=%.4f, MSE=%.4f,r^2=%.4f,cd58_spearmanr=%.4f,cd81_spearmanr=%.4f' % (mean_error, msev,r_value ** 2,cd58_sr,cd81_sr))
    print('MSE '+str(msev))
    print('R squared '+ str(r_value ** 2))
    print('cd58_spearmanr'+str(cd58_sr))
    print('cd81_spearmanr'+str(cd81_sr))
    #print('spearmanr '+str(sr))
    #print('spearmanr_p '+str(srp))

    if save:
        plt.legend()
        plt.xlabel('relative ratio (scaled)')
        plt.ylabel('predicted ratio')
        save_to_results(model_name, dataset_name, regre, kfold, split, 'test-prediction-scatter')



def get_pseudo_roc_for_regression(model: keras.Model, test_set: tf.data.Dataset, regre, kfold, split, model_name='', dataset_name='',
                                  save=True):
    plt.clf()
    outputs, labels = get_outputs_and_labels(model, test_set, sigmoid=False, invert=True)

    decision_points = 1 - np.array([0.05, 0.1, 0.2, 0.5])

    for decision_point in decision_points:
        new_true_labels = labels >= decision_point
        score = roc_auc_score(new_true_labels, outputs)
        fpr, tpr, thresholds = roc_curve(new_true_labels, outputs)
        if decision_point == 0.8: #top20%
            print('AUROC '+str(score))
        plt.plot(fpr, tpr, label='%s, decision_point=%f, AUC=%f' % (model_name, decision_point, score))
    if save:
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.title('Test Set pseudo-ROC curves')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        save_to_results(model_name, dataset_name, regre, kfold, split, 'pseudo-roc')

    plt.clf()
    for decision_point in decision_points:
        new_true_labels = labels >= decision_point
        fpr2, tpr2, threshold2 = precision_recall_curve(new_true_labels, outputs)
        score2 = average_precision_score(new_true_labels, outputs)
        if decision_point == 0.8: #top20%
            print('AUPRC '+str(score2))
        plt.plot(fpr2, tpr2, label='%s, decision_point=%f, Avg Prec=%f' % (model_name, decision_point, score2))
    if save:
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.title('Test Set Precision-Recall curves')
        plt.xlabel('precision')
        plt.ylabel('recall')
        save_to_results(model_name, dataset_name, regre, kfold, split, 'precision-recall')


def get_gradients(model, model_input):
    with tf.GradientTape() as tape:
        tape.watch(model_input)
        preds = model(model_input)

    grads = tape.gradient(preds, model_input)
    return grads


def get_integrated_gradients(model, model_input, baseline=None, num_steps=50):
    if baseline is None:
        baseline_sequence = np.zeros_like(model_input[0])
        baseline_features = np.zeros_like(model_input[1])
        #average the input as baseline
        baseline_features[:, 0] = 0.04986 #linearfold_vals
        #average gradient for position flags
        #baseline_features[:, 1] = 0.07129 # 5' utr
        #baseline_features[:, 2] = 0.637577 # cds
        #baseline_features[:, 3] = 0.291133 # 3' utr
        baseline_features[:, 4] = 0.936664 # target transcript percent
        baseline_features[:, 5] = 0.122166 # target unfold energy
        #baseline_features[:, 6] = 0.035566 # 5' utr pos floats
        #baseline_features[:, 7] = 0.319753 # cds pos floats
        #baseline_features[:, 8] = 0.139980 # 3' utr pos floats

        baseline = (baseline_sequence, baseline_features)
    else:
        baseline_sequence = baseline[0]
        baseline_features = baseline[1]

    # 1. Do interpolation.
    input_sequence, input_features = model_input
    interpolated_sequences = [
        baseline_sequence + (step / num_steps) * (input_sequence - baseline_sequence)
        for step in range(num_steps + 1)
    ]
    interpolated_sequences = np.array(interpolated_sequences).astype(np.float64)  # (51, 10, 30, 7)
    interpolated_features = [
        baseline_features + (step / num_steps) * (input_features - baseline_features)
        for step in range(num_steps + 1)
    ]
    interpolated_features = np.array(interpolated_features).astype(np.float64)  # (51, 10, 7)

    # 3. Get the gradients
    grads_sequences = []
    grads_features = []
    for i in range(num_steps + 1):
        input_s = tf.convert_to_tensor(interpolated_sequences[i], dtype=tf.float64)
        input_f = tf.convert_to_tensor(interpolated_features[i], dtype=tf.float64)
        grads_s, grads_f = get_gradients(model, (input_s, input_f))  #TensorShape([10, 50, 8]), TensorShape([10, 7])
        grads_sequences.append(grads_s) 
        grads_features.append(grads_f) 
    grads_sequences = tf.convert_to_tensor(grads_sequences, dtype=tf.float64)  # TensorShape([51, 10, 50, 8])
    grads_features = tf.convert_to_tensor(grads_features, dtype=tf.float64)  # TensorShape([51, 10, 7])

    # 4. Approximate the integral using the trapezoidal rule
    grads_sequences = (grads_sequences[:-1] + grads_sequences[1:]) / 2.0  # TensorShape([50, 10, 50, 8])
    avg_grads_sequences = tf.reduce_mean(grads_sequences, axis=0)  #  TensorShape([10, 50, 8])

    grads_features = (grads_features[:-1] + grads_features[1:]) / 2.0  #  TensorShape([51, 10, 7])
    avg_grads_features = tf.reduce_mean(grads_features, axis=0)  # TensorShape([10, 7])

    # 5. Calculate integrated gradients and return
    baseline_sequence = tf.convert_to_tensor(baseline_sequence, dtype=tf.float64)
    input_sequence = tf.cast(input_sequence, tf.float64)
    baseline_features = tf.convert_to_tensor(baseline_features, dtype=tf.float64)

    integrated_grads_sequences = (input_sequence - baseline_sequence) * avg_grads_sequences  # TensorShape([10, 50, 8])
    integrated_grads_features = (input_features - baseline_features) * avg_grads_features  # TensorShape([10, 7])
    return (integrated_grads_sequences, integrated_grads_features, avg_grads_sequences)


def random_baseline_integrated_gradients(model, model_input, baseline, num_steps=50, num_runs=1):
    integrated_grads_sequences = []
    integrated_grads_features = []
    integrated_grads_sequences_hypothetical=[]

    # 2. Get the integrated gradients for all the baselines
    for run in range(num_runs):
        igrads = get_integrated_gradients(model, model_input, baseline, num_steps)

        integrated_grads_sequences.append(igrads[0])
        integrated_grads_sequences_hypothetical.append(igrads[2])
        integrated_grads_features.append(igrads[1])

    # 3. Return the average integrated gradients 
    integrated_grads_sequences = tf.convert_to_tensor(integrated_grads_sequences)  # TensorShape([2, 10, 30, 7])
    integrated_grads_sequences = tf.reduce_mean(integrated_grads_sequences, axis=0)  # TensorShape([10, 30, 7])
    integrated_grads_features = tf.convert_to_tensor(integrated_grads_features)  # TensorShape([2, 10, 7])
    integrated_grads_features = tf.reduce_mean(integrated_grads_features, axis=0)  ##TensorShape([10, 7])
    integrated_grads_sequences_hypothetical = tf.convert_to_tensor(integrated_grads_sequences_hypothetical)
    integrated_grads_sequences_hypothetical = tf.reduce_mean(integrated_grads_sequences_hypothetical, axis=0)

    return (integrated_grads_sequences, integrated_grads_features,integrated_grads_sequences_hypothetical)

def integrated_gradients(model, test_dataset, regre, kfold, split, model_name, dataset_name):
    np.set_printoptions(suppress=True)

    np.random.seed(0)

    test_inputs = [inputs for (inputs, label) in test_dataset.unbatch()]
    test_outputs = [label for (inputs, label) in test_dataset.unbatch()]
    test_sequences = [sequences for (sequences, features) in test_inputs]
    test_features = [features for (sequences, features) in test_inputs]
    

    new_array = np.zeros((len(test_sequences), test_sequences[0].shape[0], test_sequences[0].shape[1]))
    np.copyto(new_array, test_sequences)
    test_sequences = new_array

    outputs, labels = get_outputs_and_labels(model, test_dataset, sigmoid=True)
    #rounded_outputs = np.rint(outputs)

    num_samples = len(test_inputs) #all test set
    samples = np.random.choice(len(test_inputs), num_samples, replace = False)
    #samples = np.random.choice(len(test_inputs), len(test_inputs), replace = False)

    test_sequences = tf.convert_to_tensor([test_sequences[i] for i in samples])   # TensorShape([9928, 30, 7])
    test_features = tf.convert_to_tensor([test_features[i] for i in samples])   # TensorShape([9928, 7])
    sample_inputs = (test_sequences, test_features)
    grads = get_gradients(model, sample_inputs)  # TensorShape([10, 30, 7]), TensorShape([10, 7])
    baseline = None
    igrads = random_baseline_integrated_gradients(model, sample_inputs, baseline)

    sequences_igrads, features_igrads, sequences_igrads_hypo = igrads[0].numpy(), igrads[1].numpy(), igrads[2].numpy()
    model_outputs = tf.sigmoid(model(sample_inputs)).numpy().squeeze()
    test_sequences = test_sequences.numpy()

    #store scores to .h5
    dataset_folder = RESULTS_DIR + dataset_name + '/'
    if regre:
        model_folder = dataset_folder + model_name + '_regression' + '/'
    else:
        model_folder = dataset_folder + model_name + '_classification' + '/'

    f = h5py.File('%s%s_%sof%s.h5' % (model_folder, "gradient_scores", split, kfold))
    g = f.create_group("contrib_scores")
    g.create_dataset("task0", data=sequences_igrads)
    g = f.create_group("hyp_contrib_scores")
    g.create_dataset("task0", data=sequences_igrads_hypo)
    f.close()

    # def encoded_nuc_to_str(encoded_seq):
    #     indices = np.argmax(encoded_seq, axis=1)
    #     return ''.join([base_positions[i] for i in indices])

    def encoded_linfold_to_str(encoded_seq):
        indices = np.argmax(encoded_seq, axis=1)
        zeros = np.sum(encoded_seq, axis = 1).astype('int')
        indices[zeros == 0] = -1
        return ''.join([linearfold_positions[i] for i in indices])

    print("SEQUENCE INTEGRATED GRADIENTS")
    print("min and max gradients:")
    print(sequences_igrads.min())
    print(sequences_igrads.max())
    #for i in range(num_samples):
        #print("SAMPLE " + str(samples[i]))
        #print("true_label = " + str(int(test_outputs[samples[i]].numpy())))
        #print("predicted value = " + str(model_outputs[i]))
        # print("test sequences:")
        # print(test_sequences[i])
        # print("sequences integrated gradients")
        # print(sequences_igrads[i])
        #nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
        #flags = test_sequences[i][:,4]
        #linfold_sequence = encoded_linfold_to_str(test_sequences[i][:,4:])
        #print("[nucleotide, (flag), linearfold] ----------  [nucleotide gradient, (flag gradient), linearfold gradient]")
        #for j in range(len(test_sequences[i])):
        #    sequence_list = []
        #    sequence_list.append(nuc_sequence[j])
        #    sequence_list.append(linfold_sequence[j])

        #    nonzero_indices = np.nonzero(sequences_igrads[i][j])[0]
        #    igradients = sequences_igrads[i][j][nonzero_indices]
        #    print(str(sequence_list) + " ---------- " + str(igradients))


        #print("\n")
    print("FEATURE INTEGRATED GRADIENTS")
    print("min and max gradients:")
    print(features_igrads.min())
    print(features_igrads.max())
    #for i in range(num_samples):
        #print("SAMPLE #" + str(i) + " (" + str(samples[i]) + ")")
        #print("true_label = " + str(int(test_outputs[samples[i]].numpy())))
        #print("predicted value = " + str(model_outputs[i]))
        #print("test features:")
        #print(test_features[i].numpy())
        #print("feature integrated gradients")
        #print(features_igrads[i])
        #print("\n")
    
    ###############
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.1,0.2,0.8,0.7])
    nuc_a = []
    nuc_c = []
    nuc_t = []
    nuc_g = []
    for k in range(test_sequences[0].shape[0]):
        nuc_a_average = []
        nuc_c_average = []
        nuc_t_average = []
        nuc_g_average = []
        for i in range(num_samples):
            nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
            #linfold_sequence = encoded_linfold_to_str(test_sequences[i][:,4:])
            feature_ig = features_igrads[i]
            nonzero_indices = np.nonzero(sequences_igrads[i])
            #sequence_ig = sequences_igrads[i][nonzero_indices].reshape(len(nuc_sequence), 2)
            sequence_ig = sequences_igrads[i][nonzero_indices].reshape(len(nuc_sequence), 1)


            #nuc_ig = [i[0] for i in sequence_ig]
            nuc_ig = [j for j in sequence_ig]
            #lin_ig = [i[1] for i in sequence_ig]

            if nuc_sequence[k] == "A":
                nuc_a_average.append(nuc_ig[k])
            elif nuc_sequence[k] == "C":
                nuc_c_average.append(nuc_ig[k])
            elif nuc_sequence[k] == "T":
                nuc_t_average.append(nuc_ig[k])
            else: 
                nuc_g_average.append(nuc_ig[k])

        if len(nuc_a_average) != 0:
            nuc_a_average = sum(nuc_a_average) / len(nuc_a_average)
        else:
            nuc_a_average = np.nan
        if len(nuc_c_average) != 0:
            nuc_c_average = sum(nuc_c_average) / len(nuc_c_average)
        else:
            nuc_c_average = np.nan
        if len(nuc_t_average) != 0:
            nuc_t_average = sum(nuc_t_average) / len(nuc_t_average)
        else:
            nuc_t_average = np.nan
        if len(nuc_g_average) != 0:
            nuc_g_average = sum(nuc_g_average) / len(nuc_g_average)
        else:
            nuc_g_average = np.nan

        nuc_a.append(nuc_a_average)
        nuc_c.append(nuc_c_average)
        nuc_t.append(nuc_t_average)
        nuc_g.append(nuc_g_average)

    x1 = np.arange(1, (len(nuc_sequence)+1))
    ax.set_xticks(x1)
    ax.plot(x1, nuc_a, label='A')
    ax.plot(x1, nuc_c, label='C')
    ax.plot(x1, nuc_t, label='T')
    ax.plot(x1, nuc_g, label='G')
    ax.set_title("Integrated gradients: positional nucleotide gradient")
    ax.set_xlabel('position on guide')
    #ax.set_xticklabels(sample1_labels)
    ax.set_ylabel('gradient values')
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='x', which='minor', labelsize=8)
    ax.legend()

    save_to_results(model_name, dataset_name, regre, kfold, split, 'integrated_gradients_by_nucleotide')

    # ig dataframe
    data = []
    for i in range(num_samples):
        new_row = []
        nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
        #linfold_sequence = encoded_linfold_to_str(test_sequences[i][:,4:])
        feature_ig = features_igrads[i]
        nonzero_indices = np.nonzero(sequences_igrads[i])
        #sequence_ig = sequences_igrads[i][nonzero_indices].reshape(len(nuc_sequence), 2)
        sequence_ig = sequences_igrads[i][nonzero_indices].reshape(len(nuc_sequence), 1)

        nuc_ig = [j[0] for j in sequence_ig]
        #nuc_ig = [i[0] for i in sequence_ig]
        #lin_ig = [i[1] for i in sequence_ig]

        new_row.append(nuc_sequence) #nuc sequence
        #new_row.append(linfold_sequence) #linfold sequence
        new_row = new_row + list(test_features[i].numpy()) #single value inputs
        new_row.append(test_outputs[samples[i]].numpy()) #true value
        if regre:
            model_outputs = model(sample_inputs).numpy().squeeze()
        new_row.append(model_outputs[i]) #predicted value
        
        new_row = new_row + nuc_ig #add nucleotide gradients
        #new_row = new_row + lin_ig #add linearfold gradients
        new_row = new_row + list(feature_ig) #add single value gradients

        data.append(new_row)

    nuc_grad_labels = ["nucleotide_gradient_pos" + str(i+1) for i in range(test_sequences[i].shape[0])]
    #lin_grad_labels = ["linearfold_gradient_" + str(i) for i in range(test_sequences[i].shape[0])]
    single_value_inputs = ['guide MFE','is_5UTR', 'is_CDS', 'is_3UTR','target_transcript_percent','target unfold energy','UTR5_position','CDS_position','UTR3_position']

    single_value_grads = [single_value_inputs[i] + "_gradient" for i in range(len(single_value_inputs))]

    new_df = pd.DataFrame(data, columns = ['nucleotide sequence'] 
        + single_value_inputs + ["true value", "predicted_value"] + nuc_grad_labels + single_value_grads) 

    # dataset_folder = RESULTS_DIR + dataset_name + '/'
    ## model_folder = dataset_folder + model_name + '/'
    
    # if regre:
    # 	model_folder = dataset_folder + model_name + '_regression' + '/'
    # else:
    # 	model_folder = dataset_folder + model_name + '_classification' + '/'

    if kfold == None:
        new_df.to_csv('%s%s.csv' % (model_folder, "int_grad_statistics"))
    else:
        new_df.to_csv('%s%s_%sof%s.csv' % (model_folder, "int_grad_statistics_new_baseline_gradient", split, kfold))

    #single value feature ig plot
    #df_features_ig = new_df.iloc[:,(-len(single_value_inputs)):] #feature ig
    df_features = new_df[single_value_inputs] #feature input
    df_features_ig = new_df[single_value_grads] #feature ig

    ff_ig_mean =[]
    #ff_ig_sd =[]

    for i in range(len(single_value_inputs)): #feature number
        f_name = single_value_inputs[i]
        f_input = df_features[f_name].values
        f_gradient = df_features_ig[single_value_grads[i]].values
        #f_slope=np.divide(f_gradient,f_input)
        #feature_ig_mean = np.nanmean(f_slope)
        #feature_ig_sd = np.nanstd(f_slope)
        res = stats.linregress(f_input, f_gradient) #Perform the linear regression
        #print(f"R-squared: {res.rvalue**2:.6f}") #Coefficient of determination (R-squared)
        #print('slope '+str(res.slope)) 
        feature_ig_mean = res.slope

        ff_ig_mean.append(feature_ig_mean)
        #ff_ig_sd.append(feature_ig_sd)

    plt.clf()
    fig = plt.figure(figsize = (15, 8))
    #ax = fig.add_axes([0.1,0.2,0.8,0.7])
    #fx = np.arange(len(ff_ig_mean))
    #ax.set_xticks(fx)
    
    #ax.set_title("Test guides: Single Value Inputs")
    #ax.set_xlabel('single value inputs')
    #plt.bar(single_value_inputs,ff_ig_mean, width = 0.4,yerr =ff_ig_sd)
    plt.bar(single_value_inputs,ff_ig_mean, width = 0.4)
    plt.xticks(rotation = 45)
    #ax.set_xticklabels(single_value_inputs)
    #ax.set_ylabel('gradient values')
    #ax.tick_params(axis='x', which='major', labelsize=10)
    #ax.tick_params(axis='x', which='minor', labelsize=8)
    #ax.legend()
    plt.xlabel("Secondary features") 
    plt.ylabel("Feature importance - gradient values") 
    plt.title("Gradients for Single Value Inputs") 

    save_to_results(model_name, dataset_name, regre, kfold, split, 'integrated_gradients_test_all_features_new_baseline')


    ###########
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.1,0.2,0.8,0.7])
    has_plot = False
    #nuc_ig_alltest = [ [] for _ in range(30) ]
    nuc_ig_alltest = []
    nuc_ig_alltest_abs = []
    for i in range(num_samples):
        nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
        #linfold_sequence = encoded_linfold_to_str(test_sequences[i][:,4:])
        feature_ig = features_igrads[i]
        nonzero_indices = np.nonzero(sequences_igrads[i])
        #sequence_ig = sequences_igrads[i][nonzero_indices].reshape(len(nuc_sequence), 2)
        sequence_ig = sequences_igrads[i][nonzero_indices].reshape(len(nuc_sequence), 1)

        #nuc_ig = [i[0] for i in sequence_ig]
        nuc_ig = [j[0] for j in sequence_ig]
        nuc_ig_abs = [abs(j[0]) for j in sequence_ig]
        #lin_ig = [i[1] for i in sequence_ig]
        
        nuc_ig_alltest.append(nuc_ig)
        nuc_ig_alltest_abs.append(nuc_ig_abs) #abs of nuc_ig

        #for p in range(len(nuc_sequence)):
        #    nuc_ig_alltest[p].append(sequence_ig[p])

        #x1 = np.arange(len(nuc_sequence))
        #x2 = np.arange(len(feature_ig))
        #ax.set_xticks(x1)
        #if not has_plot:
        #    ax.scatter(x1, nuc_ig, color = "r", marker=".", alpha=0.5, label='sequence')
            #ax.scatter(x1, lin_ig, color = "b", marker=".", alpha=0.5, label='linearfold')
        #    has_plot = True
        #else:
        #    ax.scatter(x1, nuc_ig, color = "r", marker=".", alpha=0.5,label = None)
            #ax.scatter(x1, lin_ig, color = "b", marker=".", alpha=0.5,label = None)

    #df_nuc_ig_all = pd.DataFrame(nuc_ig_alltest, columns = range(30))
    df_nuc_ig_all_abs = pd.DataFrame(nuc_ig_alltest_abs, columns = np.arange(1,31))
    ax = sn.boxplot(data=df_nuc_ig_all_abs) 

    ax.set_title("Integrated gradient of test guide sequence inputs")
    ax.set_xlabel('position on guide')
    #ax.set_xticklabels(sample1_labels)
    ax.set_ylabel('gradient values')
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='x', which='minor', labelsize=8)
    ax.legend()

    save_to_results(model_name, dataset_name, regre, kfold, split, 'integrated_gradients_test_all_sequences')
    
    # true positive
    # plt.clf()
    # fig = plt.figure()
    # ax = fig.add_axes([0.1,0.2,0.8,0.7])
    # has_plot = False
    # for i in range(num_samples):
    #     if samples[i] in true_positives:
    #         nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
    #         #linfold_sequence = encoded_linfold_to_str(test_sequences[i][:,4:])
    #         feature_ig = features_igrads[i]
    #         nonzero_indices = np.nonzero(sequences_igrads[i])
    #         #sequence_ig = sequences_igrads[i][nonzero_indices].reshape(len(nuc_sequence), 2)
    #         sequence_ig = sequences_igrads[i][nonzero_indices].reshape(len(nuc_sequence), 1)

    #         #nuc_ig = [i[0] for i in sequence_ig]
    #         nuc_ig = [i for i in sequence_ig]
    #         #lin_ig = [i[1] for i in sequence_ig]

    #         x1 = np.arange(len(nuc_sequence))
    #         x2 = np.arange(len(feature_ig))
    #         ax.set_xticks(x1)
    #         if not has_plot:
    #             ax.scatter(x1, nuc_ig, color = "r", marker=".", alpha=0.5,label='sequence')
    #             #ax.scatter(x1, lin_ig, color = "b", marker=".", alpha=0.5,label='linearfold')
    #             has_plot = True
    #         else:
    #             ax.scatter(x1, nuc_ig, color = "r", marker=".", alpha=0.5, label = None)
    #             #ax.scatter(x1, lin_ig, color = "b", marker=".", alpha=0.5, label = None)

    # ax.set_title("True Positives: Sequence inputs")
    # ax.set_xlabel('nucleotides')
    # #ax.set_xticklabels(sample1_labels)
    # ax.set_ylabel('gradient values')
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='x', which='minor', labelsize=8)
    # ax.legend()

    #save_to_results(model_name, dataset_name, regre, kfold, split, 'integrated_gradients_true_positives_sequences')


    ################
    #feature ig scatter plot 
    # plt.clf()
    # fig = plt.figure()
    # ax = fig.add_axes([0.1,0.2,0.8,0.7])
    # has_plot = False
    # for i in range(num_samples):
    #     nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
    #     #linfold_sequence = encoded_linfold_to_str(test_sequences[i][:,4:])
    #     feature_ig = features_igrads[i]
    #     #nonzero_indices = np.nonzero(sequences_igrads[i])
    #     #sequence_ig = sequences_igrads[i][nonzero_indices].reshape(len(nuc_sequence), 2)

    #     #nuc_ig = [i[0] for i in sequence_ig]
    #     #lin_ig = [i[1] for i in sequence_ig]

    #     x1 = np.arange(len(nuc_sequence))
    #     x2 = np.arange(len(feature_ig))
    #     ax.set_xticks(x2)
    #     if not has_plot:
    #         ax.scatter(x2, feature_ig, color = "r", marker=".", label='single value inputs')
    #         has_plot = True
    #     else:
    #         ax.scatter(x2, feature_ig, color = "r", marker=".", label = None)

    # #single_value_inputs = ['is_5UTR', 'is_CDS', 'is_3UTR', 'UTR5_position','CDS_position','UTR3_position','RNAseq relative', 'linearfold value']
    # single_value_inputs = ['linf_contra_val (MFE)','is_5UTR', 'is_CDS', 'is_3UTR','target_transcript_percent']
    # ax.set_title("Test guides: Single Value Inputs")
    # ax.set_xlabel('single value inputs')
    # ax.set_xticklabels(single_value_inputs)
    # ax.set_ylabel('gradient values')
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='x', which='minor', labelsize=8)
    # ax.legend()

    #save_to_results(model_name, dataset_name, regre, kfold, split, 'integrated_gradients_test_all_features')
