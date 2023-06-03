import argparse
from random import choice

from run import train
from run import prune
from run import prune_edges
from run import module_detection

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modularity Detection')

    parser.add_argument('--experiment', type=str, default='module_detection',
                        choices=['train', 
                                'prune',
                                'prune_edges',
                                'module_detection'])
    
    ####################################################################################################################
    # Specify Task, Function, Path to dataset
    ####################################################################################################################

    parser.add_argument('--task', type=str, default='boolean',choices=['boolean', 'mnist', 'mnist_10way'])
    parser.add_argument('--modularity', nargs='+', type=int, default= [4,2,4])
    parser.add_argument('--dataset_path', type=str, default='./datasets/exemplar/')
    parser.add_argument('--result_path', type=str, default='./results/exemplar/unit_edge/')
    parser.add_argument('--dataset_noise', type=bool, default=True)

    ####################################################################################################################
    # Specify ANN architecture and training hyper-parameters
    ####################################################################################################################

    parser.add_argument('--model', type=str, default='MLP',
                        choices=['MLP'])
    parser.add_argument('--arch', nargs='+', type=int, default=[4,24,4])
    parser.add_argument('--hidden_layer_activation', type=str, default = 'relu',
                        choices = ['relu', 'sigmoid','ste'])
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'momentum', 'rms'])
    parser.add_argument('--loss', type=str, default='BCE',
                        choices=['CE', 'MSE', 'BCE'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--bn', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--scheduler', type=str, default = 'exp', 
                        choices = ['exp', 'cosine'])
    parser.add_argument('--gamma', type=float, default = 1.0)
    

    ####################################################################################################################
    # Specify unit pruning criterion and method
    ####################################################################################################################

    parser.add_argument('--prune_acc_threshold', type=float, default=100.0)

    parser.add_argument('--prune_units', type=int, default=0)
    parser.add_argument('--unit_pruning_metric', type=str, default='activation_gradient_product',
                        choices=['average_activation_weight_magnitude', 'activation_gradient_product', 'activation_sum_gradient_product'])
    parser.add_argument('--unit_pruning_metric_normalized', type=int, default=1)
    parser.add_argument('--unit_pruning_norm', type=str, default='None',
                        choices=['mean_std', 'mean', 'std', 'sum_scores', 'prev_num_units', 'None', 'num_units'])
    parser.add_argument('--unit_pruning_scale', type=str, default='global',
                        choices=['global', 'local', 'local_global', 'None'])
    parser.add_argument('--unit_pruning_schedule', type=str, default='linear',
                        choices=['linear', 'lth', 'exponential', 'cosine', 'cubic', 'None'])
    parser.add_argument('--unit_pruning_gradient', type=float, default=None)
    parser.add_argument('--unit_pruning_iterations', type = int, default = None)

    ####################################################################################################################
    # Specify edge pruning criterion and method
    ####################################################################################################################

    parser.add_argument('--pruning_schedule', type=str, default='linear',
                        choices=['linear', 'lth', 'exp', 'cosine', 'cubic', 'None'])
    parser.add_argument('--pruning_gradient', nargs='+', type=float, default=[2.5,2.0,1.5,1.0,0.5])
    parser.add_argument('--which_unit_pruning_gradient', type=int, default=0)
    parser.add_argument('--pruning_iterations', type = int, default = None)

    ####################################################################################################################
    # Specify module detection hyper-parameters
    ####################################################################################################################

    #### Get networks and masks

    parser.add_argument('--visualization_path', type=str, default='./visualizations/exemplar/')
    parser.add_argument('--prune_perc', type=float, default = None)

    ######## Feature Vector arguments

    parser.add_argument('--feature_direction', type=str, default='outgoing', 
                        choices = ['outgoing', 'incomming', 'bidirectional'])
    parser.add_argument('--feature_weighted', type = bool, default = False)
    parser.add_argument('--feature_step_size', type = int, default = 2)
    parser.add_argument('--clusterability_step_size', type = int, default = 2)
    parser.add_argument('--hp', type = int, default = 0)

    ######## Clustering arguments

    parser.add_argument('--clusterability_criterion', type=str, default = 'compare_expected')
    parser.add_argument('--clustering_linkage', type=str, default='average', choices=['complete', 'average', 'single', 'ward'])
    parser.add_argument('--clustering_distance_measure', type=str, default='cosine', choices = ['cosine', 'jaccard', 'euclidean'])
    parser.add_argument('--stopping_criteria', type=str, default='modularity-metric', choices=['ch-index', 'silhouette', 'db-index', 'modularity-metric'])

    parser.add_argument('--cluster_merging_threshold', type=float, default = 0.9)
    parser.add_argument('--mm_threshold', type=float, default = 0.2)

    ####################################################################################################################
    # Specify runtime hyper-parameters
    ####################################################################################################################

    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.experiment == 'train':
        train.run(args)

    if args.experiment == 'prune':
        prune.run(args)

    if args.experiment == 'prune_edges':
        prune_edges.run(args)

    if args.experiment == 'module_detection':
        module_detection.run(args)
