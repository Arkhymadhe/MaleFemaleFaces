""" Train model from command line. """

import argparse
from copy import deepcopy
#from torchinfo import summary

from data_ops import *
from utils import *
from viz_utils import *

from dcgan import *

import random
import numpy as np
import gc
import shutil

try:
    from jupyterthemes import jtplot
except ModuleNotFoundError:
    pass


def main():
    ### CLI arguments
    args = argparse.ArgumentParser(description='Arguments for training a DCGAN.')

    args.add_argument('--data_dir', type=str, help='Data directory',
                      default=os.getcwd().replace('scripts', 'dataset'))

    args.add_argument('--image_dir', type=str, help='Directory for generated images',
                      default=os.getcwd().replace('scripts', 'Generated_images'))

    args.add_argument('--image_size', type=int, default=64, help='Input image shape')

    args.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args.add_argument('--stats', type=tuple, default=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      help='Image normalization statistics')

    args.add_argument('--save', default=True, type=bool, help='Save trained model')

    args.add_argument('--lr', type=float, default=2e-4, help='Convergence rate')

    args.add_argument('--beta1', type=float, default=0.5, help='First moment')

    args.add_argument('--beta2', type=float, default=0.99, help='Second moment')

    args.add_argument('--epochs', type=int, default=50, help='Number of GAN training cycles')

    args.add_argument('--decay_rate', type=float, default=0.001, help='Weight decay factor')

    args.add_argument('--figsize', type=tuple[int, int], default=(5, 5), help='Generated image figure size')

    args.add_argument('--style', type=str, default='gruvboxd',
                      choices=['gruvboxd', 'onedork', 'oceans16', 'solarizedd'],
                      help='Visualization style')

    args.add_argument('--gpu', type=bool, default=True, help='Train on GPU or not')

    args = args.parse_args()

    try:
        jtplot.style(args.style)
    except NameError:
        pass

    start_time = time.time()
    origin_time = deepcopy(start_time)

    print('>>> Parsing CLI arguments...')
    start_time = time.time()
    print(f'>>> CLI arguments parsed! Time elapsed : {keep_time(start_time):.5f} secs.')
    print()

    ### Dataset
    print('>>> Importing dataset...')
    start_time = time.time()

    data = load_data(path=args.data_dir, batch_size=args.batch_size, size=args.image_size, stats=args.stats)

    print(f'>>> Dataset successfully imported! Time elapsed : {keep_time(start_time):.5f} secs.\n')
    print()

    ### Randomly obtain a sample batch of images
    b = next(iter(data))[0]

    ### Display selected sample
    display_images(images=denorm(b.permute(0, 2, 3, 1)), nrows=4, ncolumns=4, figsize=args.figsize)

    ### Reproducibility
    print('>>> Ensuring reproducibility...')
    print('>>> Setting global and local random seeds...')

    start_time = time.time()

    random.seed(2022)
    os.environ['PYTHONHASHSEED'] = '2022'
    np.random.default_rng(2022)

    print('>>> Random seeds set!')
    print(f'>>> Reproducibility ensured! Time elapsed : {keep_time(start_time):.5f} secs.\n')

    print('>>> Construct model architectures...')
    start_time = time.time()

    device = "cuda" if args.gpu else "cpu"
    try:
        device = torch.device(device)
        print(f"Cuda device selected!\n")
    except:
        print(f"No Cuda device detected! Switching to CPU...\n")
        device = torch.device(device)

    ### Instantiate GAN object

    print(f'>>> Model & optimizer objects instantiated! Time elapsed : {keep_time(start_time):.5f} secs.\n')

    del b
    gc.collect()
    history = dict()

    ### Train GAN
    history, discriminator, generator, opt_d, opt_g = train_loop(
        data, args.epochs, folder=args.image_dir,
        lr=args.lr, betas=(args.beta1, args.beta2), decay_rate=args.decay_rate,
        criterion=nn.BCELoss(), device=device, figsize=args.figsize, history=history)

    ### Final history of metrics
    visualize_losses(history, epochs=args.epochs)

    if args.save:
        if not os.path.exists(os.getcwd().replace('scripts', 'artefacts')):
            os.makedirs(os.getcwd().replace('scripts', 'artefacts'))

        gen_name = "saved_generator.ckpt"
        dis_name = "saved_discriminator.ckpt"

        print('>>> Saving model weights...')
        torch.save(
            {
                'model_state_dict': generator.state_dict(),
                'opt_state_dict': opt_g.state_dict()
            },
            gen_name
        )

        torch.save(
            {
                'model_state_dict': discriminator.state_dict(),
                'opt_state_dict': opt_d.state_dict()
            },
            dis_name
        )

        ### Move trained artefacts to appropriate directories
        shutil.move(dis_name,
                    os.path.join(os.getcwd().replace("scripts", "artefacts"), dis_name))
        shutil.move(gen_name,
                    os.path.join(os.getcwd().replace("scripts", "artefacts"), gen_name))

        print('>>> Weights saved!\n')

    print(f'>>> Program complete! Total time elapsed: {keep_time(origin_time)/60: .6f} mins')


### Run program
if __name__ == '__main__':
    main()
