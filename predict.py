import subprocess
import sys
import os
import tempfile
from rnatargeting.predict import predict_ensemble_test, parse_prediction_results
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
    # If input is a bytes string, write to file
    if isinstance(fpath, bytes):
        tmpfile = os.path.join('data', 'tmp.fasta')
        with open(tmpfile, 'w') as tmp:
            tmp.write(fpath.decode('utf-8'))
        fpath = tmpfile

    # Get basename of input file
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

    # return file path file for final table
    return pred_df

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



if __name__ == '__main__':
    run_pred(str(sys.argv[1]))
