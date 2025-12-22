import argparse


def parse_arguments():

    parser = argparse.ArgumentParser(description='ARGUS models')
    
    parser.add_argument('--file_path', type=str, help='path to the folder containinig the Radon data', required=True)
    parser.add_argument('--arc', default='gruse', help='network architecture: crn and gruse')
    parser.add_argument('--model_id', default='', help='Identifier of model parameter changes')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false', help='do not use cuda')
    parser.add_argument('--batch_size', type=int, default=30, metavar='N', help='Batch size to use')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--num_layers', type=int, default=3, metavar='N', help='Number of LSTM layers')
    parser.add_argument('--hidden_units', type=int, default=64, metavar='N', help='Number of neurons in hidden layer')
    parser.add_argument('--num_feats', type=int, default=1, metavar='N', help='Number of variables')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='N', help='Drop_ut value')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--epochsinfo', type=int, default=10, metavar='N', help='number of epochs to give info')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs of no loss improvement before stop training')
    parser.add_argument('--optimizer', default='adam', help='optimization method: sgd | adam | rmsprop')
    parser.add_argument('--weight_decay', type=float, default=0.001, metavar='WD', help='weight decay for the optimizer')
    parser.add_argument('--forecast', type=int, default=6, help='Forecast window')
    parser.add_argument('--lookback', type=int, default=216, help='lookback window')
    parser.add_argument('--ar', type=int, default=3, help='times repeated')
    parser.add_argument('--freeze', type=bool, default=True, help='If freeze')
    parser.add_argument('--lower_q', type=float, default=0.1, help='Lower quantile in custom loss')
    parser.add_argument('--higher_q', type=float, default=0.9, help='Higher quantile in custom loss')
    parser.add_argument('--loss_weight', type=float, default=2, help='Extremes weight in custom loss')
    parser.add_argument('--alpha', type=float, default=1.0, help='Hyperparameter of dense weight computation')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta: weight of regression loss and; 1-Beta: weight of classification loss')
    parser.add_argument('--checkpoint', default='models/checkpoints', metavar='CHECKPOINT', help='checkpoints directory')

    return parser.parse_args()
