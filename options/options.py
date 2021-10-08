import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='chooses which dataset generator to use (name of file but without _dataset)')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--saved', type=str, default=None, help='load path')
    # model parameters
    parser.add_argument('--model', required=True, type=str,
                        help='chooses which model to use (name of file but without _model)')
    parser.add_argument('--testset_path', type=str, default=None, help='testset csv path')
    parser.add_argument('--regression', '-r', help='perform regression instead of classification', action='store_true')
    parser.add_argument('--focal', '-f', help='use focal loss instead of binary cross entropy', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gradients', '-g', help='perform integrated gradients', action='store_true')
    parser.add_argument('--weighted', '-w', help='weighted loss', action='store_true')
    parser.add_argument('--kfold', '-k', type=int, help='number of kfold cross validations')
    parser.add_argument('--split', '-s', type=int, help='split of kfold')
    #feature parameters
    parser.add_argument('--guidelength', '-l', type=int, default=30, help='guide length')
    parser.add_argument('--flanklength', '-fl', type=int, default=15, help='flank length')

    return parser.parse_args()
