import tensorflow as tf
from tensorflow import keras
#import datetime
import pdb
import numpy as np
from dataset import find_dataset_generator_using_name
from models import find_model_using_name
from options.options import get_arguments
from rnatargeting.utils import *

tf.random.set_seed(0)
np.random.seed(0)

def logits_mean_absolute_error(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    return keras.metrics.mean_absolute_error(y_true, y_pred)

def logits_mean_squared_error(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    return keras.metrics.mean_squared_error(y_true, y_pred)

def wbce(y_true, y_pred, weight1 = 1, weight0 = 1) :
    y_true = tf.keras.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = tf.keras.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
    return tf.keras.mean( logloss, axis=-1)

def predict_ensemble_test(args):
    model_generator = find_model_using_name(args.model)
    dataset_generator = find_dataset_generator_using_name(args.dataset)

    model = model_generator(args)
    model: keras.Model
    test_dataset = dataset_generator(args)
    if args.saved != None:
        predict_allf = []
        for s in range(9):
            model_path = args.saved+('/fold_'+ str(s))
            if args.regression:
                model = keras.models.load_model(
                    model_path,custom_objects={
                        'logits_mean_absolute_error': logits_mean_absolute_error,
                        'logits_mean_squared_error': logits_mean_squared_error
                    }
                )
            else:
                model = keras.models.load_model(model_path)
            output = np.array(model.predict(test_dataset).flat)
            predict_allf.append(output)
        predict_allf = np.array(predict_allf)
        # mean across ensemble members
        predict_mean = np.mean(predict_allf, axis=0)
        #sigmoid
        outputs = np.array(list(tf.sigmoid(predict_mean).numpy().flat))
        test_inputs = [inputs for (inputs, label) in test_dataset.unbatch()]

        if len(test_inputs[0]) == 2:
            test_sequences = [np.array(sequences) for (sequences, features) in test_inputs]
            test_features = [features for (sequences, features) in test_inputs]
        else:
            test_sequences = [np.array(sequences[0]) for sequences in test_inputs]

        def encoded_nuc_to_str(encoded_seq):
            indices = np.argmax(encoded_seq, axis=1)
            return ''.join([base_positions[i] for i in indices])

        test_predic =[]
        for i in range(len(outputs)):
            nuc_sequence = encoded_nuc_to_str(test_sequences[i][:,0:4])
            test_predic.append([nuc_sequence,outputs[i]])
        test_df = pd.DataFrame(test_predic, columns = ['spacer sequence', 'predicted_value_sigmoid'])


        dataset_folder = 'results/' + args.dataset + '/' 
        prefix = str(args.testset_path)[:-4].split('/')[-1]
        
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)

        test_df.to_csv('%s%s_ensemble.csv' % (dataset_folder, prefix+"_guide_prediction"))
        
    print("done!")


if __name__ == '__main__':
    # Enable the following 3 lines if using a graphics card and you get CUDNN_STATUS_INTERNAL_ERROR
    args = get_arguments()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        gpus_to_use = args.gpu_ids.split(",")
        for i in range(len(gpus_to_use)):
            gpu_id = int(gpus_to_use[i])
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)

    predict_ensemble_test(args)
