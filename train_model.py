import os
import importlib
import logging
import mxnet as mx
import argparse
import common.mtcnn_metric as metric


def parse_args():
    parser = argparse.ArgumentParser(description='train mtcnn model')
    parser.add_argument('netname', type=str, help='the network name (pnet, rnet, onet)')
    parser.add_argument('--resume', dest='begin_epoch', type=int, default=0, help='resume from epoch n')
    parser.add_argument('--test', action='store_true', help='test network')
    return parser.parse_args()


def setup_logger(netname):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    head = '%(asctime)-15s %(message)s'
    formatter = logging.Formatter(head)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if not os.path.exists('log'):
        os.mkdir('log')

    log_path = 'log/%s.log' % netname
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def setup_checkpoint(netname):
    model_prefix = 'model/%s' % netname
    if not os.path.exists(model_prefix):
        os.makedirs(model_prefix)
    model_prefix += '/%s' % netname
    cb = mx.callback.do_checkpoint(model_prefix)
    return model_prefix, cb


def setup_devs(config):
    if getattr(config, 'gpus', None):
        devs = [mx.gpu(i) for i in config.gpus]
    else:
        devs = mx.cpu()
    return devs


def setup_net_args(begin_epoch=0):
    if begin_epoch:
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, begin_epoch)
    else:
        arg_params = None
        aux_params = None
    return arg_params, aux_params


def setup_basic_env(netname):
    config = importlib.import_module('config.' + netname)
    symbol = importlib.import_module('symbol.' + netname)
    dataset = importlib.import_module('dataset.' + netname)
    return config, symbol, dataset


if __name__ == '__main__':
    args = parse_args()
    logger = setup_logger(args.netname)
    config, symbol, dataset = setup_basic_env(args.netname)
    ###########################################

    net_symbol = symbol.get_symbol()
    train_params = config.get_train_params(args.begin_epoch)

    if not os.path.exists('pred_sym'):
        os.mkdir('pred_sym')
    symbol.get_symbol(False).save('pred_sym/%s-symbol.json' % args.netname)

    ###########################################
    logging.info('start with arguments:')
    logging.info(train_params)
    ###########################################

    model_prefix, checkpoint_callback = setup_checkpoint(args.netname)
    arg_params, aux_params = setup_net_args(train_params.get('begin_epoch', 0))

    if hasattr(symbol, 'pretrained_model'):
        save_dict = mx.nd.load(symbol.pretrained_model)
        arg_params = {}
        aux_params = {}
        loaded_params = []
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            loaded_params.append(name)
            if tp == 'arg':
                arg_params[name] = v
            if tp == 'aux':
                aux_params[name] = v

        logging.info('')
        logging.info('loaded_params:')
        logging.info(loaded_params)
        # list match and unmatch params
        match_params = []
        unmatch_params = []
        for p in net_symbol.list_arguments():
            if p in arg_params or p in aux_params:
                match_params.append(p)
            else:
                unmatch_params.append(p)

        logging.info('')
        logging.info('match_params:')
        logging.info(match_params)

        logging.info('')
        logging.info('unmatch_params:')
        logging.info(unmatch_params)

    devs = setup_devs(config)
    ###########################################

    model = mx.mod.Module(context=devs, symbol=net_symbol,
                          data_names=symbol.data_names,
                          label_names=symbol.label_names,
                          logger=logger)

    eval_metrics = [metric.MtcnnAcc(), metric.MtcnnClsAcc(0), metric.MtcnnClsAcc(1), metric.MtcnnMae(), metric.MtcnnClsRatio(0)]

    train_data = dataset.get_train_iter()
    eval_data = dataset.get_test_iter()
    batch_end_callbacks = [mx.callback.Speedometer(dataset.batch_size, 500, auto_reset=False), ]
    model.fit(train_data=train_data,
              eval_data=eval_data,
              eval_metric=eval_metrics,
              batch_end_callback=batch_end_callbacks,
              epoch_end_callback=checkpoint_callback,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              **train_params)

