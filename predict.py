import subprocess
import sys
#import os



def run_pred(filepath):
	#my_env = os.environ.copy()
	#'/usr/sbin:/sbin:'
    fpath = filepath # input fasta path, should be in the "dataset" folder, and named as "dataset/\<mygenes\>.fasta"
    #prefix = ('.'.join(fpath.split('.')[:-1])).split('/')[-1] # gene name

    # Make guide library and Linearfold input:
    subprocess.run(['python','scripts/make_guide_library_features.py',fpath])

    # run Linearfold
    #subprocess.run(['cd','LinearFold'])
    # guide mfe input
    guide_l_in = '.'.join(fpath.split('.')[:-1])+"_linfold_guides_input.fasta"
    guide_l_out = '.'.join(fpath.split('.')[:-1])+"_linfold_guides_output.txt"
    p1 = subprocess.Popen(['cat', guide_l_in], stdout=subprocess.PIPE)
    fout = open(guide_l_out, 'w')
    #p2 = subprocess.run(['Linearfold/./linearfold'], stdin=p1.stdout, stdout=fout)
    p2 = subprocess.run(['LinearFold/linearfold'], stdin=p1.stdout, stdout=fout)
    # target with 15nt flanks
    target_fl_in = '.'.join(fpath.split('.')[:-1])+"_linfold_guides_nearby15_input.fasta"
    target_fl_out = '.'.join(fpath.split('.')[:-1])+"_linfold_target_flank15_output.txt"
    p3 = subprocess.Popen(['cat', target_fl_in], stdout=subprocess.PIPE)
    fout2 = open(target_fl_out, 'w')
    #p4 = subprocess.run(['Linearfold/./linearfold'], stdin=p3.stdout, stdout=fout2)
    p4 = subprocess.run(['LinearFold/linearfold'], stdin=p3.stdout, stdout=fout2)
    # target with constraints
    target_fl_c_in = '.'.join(fpath.split('.')[:-1])+"_linfold_guides_constraints_nearby15_input.fasta"
    target_fl_c_out = '.'.join(fpath.split('.')[:-1])+"_linfold_constraints_target_flank15_output.txt"
    p5 = subprocess.Popen(['cat', target_fl_c_in], stdout=subprocess.PIPE)
    fout3 = open(target_fl_c_out, 'w')
    #p6 = subprocess.run(['Linearfold/./linearfold','--constraints'], stdin=p5.stdout, stdout=fout3)
    p6 = subprocess.run(['LinearFold/linearfold','--constraints'], stdin=p5.stdout, stdout=fout3)
    #subprocess.run(['cd','..'])
    feature_f1 = '.'.join(fpath.split('.')[:-1])+'_guide_library.csv'
    subprocess.run(['python','scripts/linearfold_integrate_results.py',feature_f1])
    feature_f = '.'.join(fpath.split('.')[:-1]) + '_guides_integrated_features.csv'
    # Predict guide efficiency using the CNN model
    subprocess.run(['python3','predict_ensemble_test.py','--dataset','CNN_sequence_input','--model','guide_nolin_threef',
        '--saved','saved_model/sequence_only_input_3f','--testset_path',feature_f])
    prefix = feature_f[:-4].split('/')[-1]
    resultf = 'results/CNN_sequence_input/'+prefix+'_guide_prediction_ensemble.csv'
    #resultf = 'results/CNN_sequence_input/'+prefix+'_guides_integrated_features/test_prediction_guidelength-30_ensemble.csv'
    subprocess.run(['python','scripts/parse_prediction_results.py',feature_f1,resultf])
    outf = 'results/'+feature_f1[:-4].split('/')[-1]+'_prediction_sorted.csv'
    final_p = "static/"+outf
    subprocess.run(['mv',outf,final_p])
    subprocess.run(['rm',guide_l_in])
    subprocess.run(['rm',guide_l_out])
    subprocess.run(['rm',target_fl_in])
    subprocess.run(['rm',target_fl_out])
    subprocess.run(['rm',target_fl_c_in])
    subprocess.run(['rm',target_fl_c_out])
    subprocess.run(['rm',feature_f1])
    subprocess.run(['rm',feature_f])
    subprocess.run(['rm',resultf])
    subprocess.run(['rm',fpath])
    return final_p # return output path

    #print("done!")


# if __name__ == '__main__':
#     main(str(sys.argv[1]))
