
from pretrain import *
from utils import util

from gb_test import ModelManager


if __name__ == '__main__':
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    print('Parameters Initialization...')

    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    print('Training begin...')
    manager_p1 = PretrainModelManager(args, data)
    manager_p1.train(args, data)
    print('Training finished!')

    print('Calculate ball begin...')
    gb_centroids, gb_radii, gb_labels=manager_p1.calculate_granular_balls(args, data)
    print('Calculate ball  finished!')



    manager = ModelManager(args, data, manager_p1.model)
    print('Evaluation begin...')
    manager.evaluation(args, data, gb_centroids, gb_radii, gb_labels,mode="test")
    print('Evaluation finished!')
    util.summary_writer.close()