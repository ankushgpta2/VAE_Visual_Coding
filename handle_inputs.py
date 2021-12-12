import argparse
import tensorflow as tf


def get_hyperparams():
    args = get_args().parse_args()
    # general parameters for data
    validate_flag = args.validate
    splits = args.splits
    metrics = args.metrics
    run_vae = args.run_vae
    general_parameters = {'validate_flag': validate_flag, 'splits': splits, 'metrics': metrics, 'run_vae': run_vae}
    
    # for VAE --------------------------------------------------------->
    vae_epochs = args.vae_epochs
    vae_batch_size = args.vae_batch_size
    vae_learning_rate = args.vae_learning_rate
    vae_reg = args.vae_reg
    vae_latent_dim = args.vae_latent_dim
    vae_loss = args.vae_loss
    vae_optimizer = args.vae_optimizer
    vae_batch_norm = args.vae_batch_norm
    vae_plot_latent = args.vae_plot_latent
    if vae_loss == 'KLDivergence':
        vae_loss = tf.keras.losses.KLDivergence()
    elif vae_loss == 'sparse_categorical_crossentropy':
        vae_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    elif vae_loss == 'mse':
        vae_loss = tf.keras.losses.MeanSquaredLogarithmicError()
    vae_parameters = {'vae_epochs': vae_epochs, 'vae_batch_size': vae_batch_size, 'vae_reg': vae_reg, 'vae_learning_rate': vae_learning_rate, 'vae_loss': vae_loss,
                      'vae_optimizer': vae_optimizer, 'vae_batch_norm': vae_batch_norm, 'vae_plot_latent': vae_plot_latent, 'vae_latent_dim': vae_latent_dim}

    # for artificial --------------------------------------------------------->
    run_artificial = args.run_artificial
    artificial_info = {'run_artificial': run_artificial}
    return general_parameters, vae_parameters, artificial_info


def get_args():
    parser = argparse.ArgumentParser(description="Parameters For Neural Nets")
    # general parameters for data
    parser.add_argument('--validate', type=str, default='True', help='whether to split into validate or not')
    parser.add_argument('--splits', nargs='+', default=[0.6, 0.4], help='actual splitting data')
    parser.add_argument('--metrics', type=list, default=['accuracy'], help='accuracy metric for model')
    # which structures to run
    parser.add_argument('--run_vae', type=str, default='True', help='Whether or not to run VAE')
    # for VAE
    parser.add_argument('--vae_epochs', nargs='+', default=[100], help='number of epochs to train')
    parser.add_argument('--vae_batch_size', nargs='+', default=[200], help='batch size')
    parser.add_argument('--vae_reg', nargs='+', default=[0], help='regularization lambda')
    parser.add_argument('--vae_learning_rate', type=float, default=[0.001], help='learn rate')
    parser.add_argument('--vae_latent_dim', type=int, default=[2, 3], help='whether or not to plot latent space')
    parser.add_argument('--vae_loss', type=str, default='sparse_categorical_crossentropy', help='loss for model')
    parser.add_argument('--vae_optimizer', type=str, default='Adam', help='optimizer for loss function')
    parser.add_argument('--vae_batch_norm', type=str, default='True', help='whether or not to add batchnorm layers')
    parser.add_argument('--vae_plot_latent', type=str, default='True', help='whether or not to plot latent space')
    # for artificial data
    parser.add_argument('--run_artificial', type=bool, default=True, help='number of epochs to train')
    return parser
