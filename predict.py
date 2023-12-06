import subprocess
import sys
import os
import tempfile
from rnatargeting.predict import predict_ensemble_test
from rnatargeting.linearfold import make_guide_library_features, linearfold_integrate_results



def run_subprocess(cmd):
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

def run_linearfold(infile, outfile, params = []):
    sys.stderr.write(f"Running LinearFold on {infile}\n")
    with open(infile) as inF, open(outfile, 'w') as outF:
        subprocess.run(['LinearFold/linearfold'] + params, stdin=inF, stdout=outF)

def run_pred(fpath):
    fpath_prefix = os.path.splitext(fpath)[0]
    
    # Make guide library and Linearfold input:
    feature_files = make_guide_library_features(fpath)

    # Run LinearFold
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
    predict_ensemble_test(
        dataset_name='CNN_sequence_input', 
        model_name='guide_nolin_threef',
        saved='saved_model/sequence_only_input_3f',
        testset_path=feature_f,
        guidelength=30,
        flanklength=15
    )

    exit()
    

    # Predict guide efficiency using the CNN model
    subprocess.run(['python3','predict_ensemble_test.py','--dataset','CNN_sequence_input','--model','guide_nolin_threef',
        '--saved','saved_model/sequence_only_input_3f','--testset_path',feature_f])
    prefix = feature_f[:-4].split('/')[-1]
    resultf = 'results/CNN_sequence_input/'+prefix+'_guide_prediction_ensemble.csv'
    #resultf = 'results/CNN_sequence_input/'+prefix+'_guides_integrated_features/test_prediction_guidelength-30_ensemble.csv'
    subprocess.run(['python','scripts/parse_prediction_results.py',feature_f1,resultf])

    exit()

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


if __name__ == '__main__':
    run_pred(str(sys.argv[1]))
