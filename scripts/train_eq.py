import torch, numpy as np, yaml
import os, time, torch, numpy as np, argparse
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader 

import pkufiber.dsp.nonlinear_compensation as nl
import pkufiber.dsp.nonlinear_compensation.loss as loss_lib
from pkufiber.simulation.receiver import ber 
from pkufiber.dsp.nonlinear_compensation.loss import mse, adaptive_ber, p_mse, weight_mse, l1_l2_regularization_loss
from pkufiber.utils import qfactor

from pkufiber.data import FiberDataset, MixFiberDataset
from pkufiber.dsp.nonlinear_compensation.opt import AlterOptimizer
# from .old_loader import OldData


def write_log(writer, epoch, train_loss, metric):
    '''
    writer: SummaryWriter
    epoch: int
    train_loss: float
    metric: dict
    '''

    writer.add_scalar('Loss/train',  train_loss, epoch)
    writer.add_scalar('Loss/test',  metric['MSE'], epoch)
    writer.add_scalar('Metric/SNR', metric['SNR'], epoch)
    writer.add_scalar('Metric/BER', metric['BER'], epoch)
    writer.add_scalar('Metric/Qsq', metric['Qsq'], epoch)


def init_model(model_name, model_info):
    '''
    model_name: str
    model_info: dict
    '''
    module = getattr(nl, model_name) 
    model = module(**model_info)
    return model



def load_param(module, path='experiments/ampbc_M401/models/29.pth'):
    dic = torch.load(path, map_location='cpu')
    module.load_state_dict(dic['model_param'])
    print(f'Model {module} loaded params from:', path, flush=True)



def check_data_config(config, overlaps:int=0):
    '''
    define the window size for training and testing data.
    '''
    if config['model_name'] in ['MultiStepAMPBC', 'MultiStepPBC', 'EqFno', 
                                'EqFrePBC', 'EqFreAMPBC',  'EqAMPBCstep', 'EqPBCstep', 'EqBiLSTMstep', 'EqConvAMPBC', 'EqStftPBC','EqStftAMPBC', 'EqFreqTimePBC',
                                  'EqDBP', 'EqDBP_test', 'EqPbcDBP','EqAMPbcDBP', 'EqFreqPbcDBP', 'EqFreqAMPbcDBP',
                                  'EqDBP_trainD', 'EqPbcDBP_trainD', 'EqAMPbcDBP_trainD', 'EqFreqPbcDBP_trainD', 'EqFreqAMPbcDBP_trainD',
                                  'EqSNSEDBP', 'EqStftSnsePBC']:
        
        config['train_data']['window_size'] = config['train_data']['strides']  + overlaps
        config['test_data']['window_size'] = config['test_data']['strides']  + overlaps
        config['train_data']['Tx_window'] = True
        config['test_data']['Tx_window'] = True
    elif config['model_name'] in ['EqAMPBCaddNN', 'EqAMPBCaddFNO']:
        assert config['train_data']['strides'] == 1
        assert config['test_data']['strides'] == 1
        M = max(config['model_info']['pbc_info']['M'], config['model_info']['nn_info']['M'])
        config['train_data']['window_size'] =  M
        config['test_data']['window_size'] =  M
        config['train_data']['Tx_window'] = False
        config['test_data']['Tx_window'] = False
    elif config['model_name'] in ['EqSoPBC']:
        assert config['train_data']['strides'] == 1
        assert config['test_data']['strides'] == 1
        config['train_data']['window_size'] = config['model_info']['fo_info']['M']
        config['test_data']['window_size'] = config['model_info']['fo_info']['M']
        config['train_data']['Tx_window'] = False
        config['test_data']['Tx_window'] = False
    else:
        assert config['train_data']['strides'] == 1
        assert config['test_data']['strides'] == 1
        config['train_data']['window_size'] = config['model_info']['M']
        config['test_data']['window_size'] = config['model_info']['M']
        config['train_data']['Tx_window'] = False
        config['test_data']['Tx_window'] = False

def define_optimizer(net, config):

    # print('Use AlterOptimizer !', flush=True)
    # optimizer = AlterOptimizer([net.pbc.parameters(), net.nn.parameters()], [0, 0.001], alternate=False)

    if config['model_name'] in ['EqAMPBCaddNN', 'EqAMPBCaddFNO']:
        if 'pbc_path' not in config.keys() and 'model_path' not in config.keys():
            print('Train pbc + nn. Use different learning rate for pbc and nn !', flush=True)
            optimizer = torch.optim.Adam([{'params': net.pbc.parameters(), 'lr': config['lr']}, {'params': net.nn.parameters(), 'lr': config['lr']*10}])
        else:
            print('Train nn only.', flush=True)
            optimizer = torch.optim.Adam([{'params': net.nn.parameters(), 'lr': config['lr']*10}])
    elif config['model_name'] == 'EqFreqTimePBC':
        print('Use different learning rate for AMPBC and FreqPBC !', flush=True)
        optimizer = torch.optim.Adam([{'params': net.ampbc.parameters(), 'lr': 1e-4}, {'params': net.stftpbc.parameters(), 'lr': 1e-6}])
    else:
        print('Use same learning rate for all parameters !', flush=True)
        optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': config['lr']}])
    
    return optimizer
    

def test_model(net, dataloader, device='cuda:0'):
    '''
    net: nn.Module
    dataloader: torch.utils.data.DataLoader
    ''' 
    net.eval()
    mse_value, power_value, ber_value, N = 0,0,0, len(dataloader)

    Nbatch = 0 
    ys = []
    xs = []

    with torch.no_grad():
        for Rx, Tx, info in dataloader:
            Rx, Tx, info = Rx.to(device), Tx.to(device), info.to(device)   # [batch, window_size, Nomdes]
            PBC = net(Rx, info)                                # [batch,  Nomdes]  or [batch, window_size - overlaps,  Nomdes]
            # print(Rx.shape, Tx.shape, PBC.shape)
            assert PBC.ndim == Tx.ndim
            if Tx.ndim == 3:
                Tx = Tx[:, net.overlaps//2:Tx.shape[1]-net.overlaps//2, :]
            
            # print(Rx.shape, Tx.shape, PBC.shape)
            power_value += mse(Tx, 0).item()*Tx.shape[0]
            mse_value += mse(PBC, Tx).item()*Tx.shape[0]
            ber_value += np.mean(ber(PBC, Tx)['BER'])*Tx.shape[0]
            Nbatch += Tx.shape[0]

            ys.append(PBC.cpu().detach())
            xs.append(Tx.cpu().detach())
    power_value = power_value/Nbatch
    mse_value = mse_value/Nbatch
    ber_value = ber_value/Nbatch
    net.train()

    return {'MSE':mse_value, 'SNR': 10*np.log10(power_value/mse_value), 'BER':ber_value, 'Qsq': qfactor(ber_value)}, (torch.cat(ys, dim=0), torch.cat(xs, dim=0))



def train_model(
    writer, 
    need_weight, 
    loss_func, 
    net, 
    train_loader, 
    test_loader, 
    optimizer, 
    scheduler,  # 设置为可选
    epochs,       # 设置默认值
    model_path, 
    save_model=True, 
    save_interval=1, 
    device='cuda:0', 
    model_info=None,
    l1_lamba=1e-3,
    l2_lamba=1e-3,
):
    metric, result = test_model(net, test_loader)
    print('Test BER: %.5f, Qsq: %.5f, MSE: %.5f' % (metric['BER'], metric['Qsq'], metric['MSE']), flush=True)
    write_log(writer, 0, 0, metric)

    for epoch in range(1, epochs + 1):
        N = len(train_loader)
        train_loss = 0
        regular_loss = 0
        t0 = time.time()
        for i, (Rx, Tx, info) in enumerate(train_loader):
            # Rx,Tx, info: [B, M, Nmodes], [B,Nmodes], [B, 4]   or   [B, N, Nmodes], [B,M, Nmodes], [B, 4]
            Rx, Tx, info = Rx.to(device), Tx.to(device), info.to(device)
            PBC = net(Rx, info)
            assert PBC.ndim == Tx.ndim

            if Tx.ndim == 3:
                Tx = Tx[:, net.overlaps//2:Tx.shape[1]-net.overlaps//2, :]
                Rx = Rx[:, net.overlaps//2:Rx.shape[1]-net.overlaps//2, :]
            else:
                Rx = Rx[:, Rx.shape[1]//2, :]

            if need_weight:
                loss =  loss_func(PBC, Tx, epoch=epoch, weight=torch.abs(Rx - Tx)) 
            else:
                loss = loss_func(PBC, Tx, epoch=epoch)
            
            regular_term = l1_l2_regularization_loss(net, l1_lambda=l1_lamba, l2_lambda=l2_lamba)
            # dispersion_loss = 
            total_loss = loss + regular_term
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += loss.item()
            regular_loss += regular_term.item() 
            # print('Epoch: %d, Loss: %.5f' % (epoch, train_loss/N), flush=True)
            writer.add_scalar('Loss/train_batch', total_loss.item(), epoch*N+i)
            writer.add_scalar('Loss/fit_batch', loss.item(), epoch*N+i)
            writer.add_scalar('Loss/regular_batch', regular_term.item(), epoch*N+i)

        t1 = time.time()
        scheduler.step()
        metric, _ = test_model(net, test_loader)

        print('Epoch: %d, Loss: %.5f, Regular_term: %.5f, Time: %.5f' % (epoch, train_loss/N, regular_loss/N, t1-t0), flush=True)
        print('Test BER: %.5f, Qsq: %.5f, MSE: %.5f' % (metric['BER'], metric['Qsq'], metric['MSE']), flush=True)

        if save_model and epoch % save_interval == 0:
            ckpt = {
                'model_name': net.__class__.__name__,
                'model_info': model_info,
                'model_param': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if not os.path.exists(model_path): os.makedirs(model_path)
            torch.save(ckpt, model_path + f'/{epoch}.pth')
            print('Model saved')

        write_log(writer, epoch, train_loss/len(train_loader), metric)




def main():
    parser = argparse.ArgumentParser(description='Train DBP Model with config.')
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')
    parser.add_argument('--log_path', type=str, default='log', help='path to save logs')
    parser.add_argument('--model_path', type=str, default='model', help='path to save models')
    args = parser.parse_args()
    with open(args.config, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    torch.manual_seed(config['seed'])
    os.makedirs(args.log_path, exist_ok=True)
    writer = SummaryWriter(args.log_path)
    for k,v in config.items(): writer.add_text(k, str(v))

    print('*****  Experitment: %s *****' % args.log_path)
    print('Training Start at time: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), flush=True)

    # load model 
    net = init_model(config['model_name'], config['model_info'])

    ############################### special case #########################################
    # EqSOPBC: load pbc params and freeze pbc parameters
    if config['model_name'] == 'EqSoPBC' and 'pbc_path' in config.keys():
        load_param(net.pbc, config['pbc_path'])
        for param in net.pbc.parameters():
            param.requires_grad = False
        print('Freeze pbc parameters !', flush=True)

    if config['model_name'] == 'EqSoNN' and 'pbc_path' in config.keys():
        load_param(net.pbc, config['pbc_path'])
        for param in net.pbc.parameters():
            param.requires_grad = False
        print('EqSoNN: Load pbc params and Freeze pbc parameters !', flush=True)

    if config['model_name'] in ['EqAMPBCaddNN', 'EqSoNN'] and 'pbc_path' in config.keys():
        load_param(net.pbc, config['pbc_path'])
    
    if config['model_name'] == 'EqAMPBCaddNN' and 'nn_path' in config.keys():
        load_param(net.nn, config['nn_path'])
    
    # 冻结所有层的参数, fine-tune
    if 'opt' in config.keys() and config['opt'] == 'fine-tune':
        for param in net.nn.parameters():
            param.requires_grad = False
        # init net.nn.dense.weight to zero 
        net.nn.dense.weight.data.fill_(0)
        net.nn.dense.weight.requires_grad = True
    
    if 'model_path' in config.keys():
        load_param(net, config['model_path'])

    ############################### special case #########################################

    net = net.to(config['device'])
    # 实验发现当dtaps比较大时做不做预训练效果差不多，dtaps比较小时预训练效果更好
    if config['model_name'] in ['EqDBP_trainD', 'EqPbcDBP_trainD', 'EqAMPbcDBP_trainD', 'EqFreqPbcDBP_trainD', 'EqFreqAMPbcDBP_trainD', 'EqSNSEDBP']:
        if config['pretrainD'] == True:
            weight = config['weightD'] if 'weightD' in config.keys() else None
            net.linear.train_filter(lr=3e-4, weight=weight)    # train D-filter

        # 实验发现不冻结参数的效果更好 ！！
        if config['trainD_again'] == False:
            for param in net.linear.parameters():
                param.requires_grad = False

    # optimizer
    optimizer = define_optimizer(net, config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_step'], gamma=config['decay_gamma'])

    # load data 
    overlaps = getattr(net, 'overlaps', 0)
    check_data_config(config, overlaps=overlaps)

    train_data = MixFiberDataset(**config['train_data'])
    test_data = MixFiberDataset(**config['test_data'])
    # train_data = OldData(mode='train',window_size=41, num_symb=4000000, truncate=10000)
    # test_data = OldData(mode='test', window_size=41, num_symb=100000, truncate=10000)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=False)
    if 'test_batch_size' not in config.keys(): config['test_batch_size'] = config['batch_size']
    test_loader = DataLoader(test_data, batch_size=config['test_batch_size'], shuffle=False, drop_last=False)
    print('Train Data number:',len(train_data), 'length of train_loader:', len(train_loader))
    print('Test Data number:',len(test_data), 'length of test_loader:', len(test_loader))

    # loss function 
    need_weight = False
    if config['loss_type'] == 'weight_mse':
        from functools import partial
        loss_func = partial(weight_mse, p=config['p'])
        need_weight = True
    elif config['loss_type'] == 'MSE':
        loss_func = mse
    elif config['loss_type'] == 'p_mse':
        from functools import partial
        loss_func = partial(p_mse, p=config['p'])
    else:
        loss_func = getattr(loss_lib, config['loss_type'])
    
    if 'l1_lambda' not in config.keys(): config['l1_lambda'] = 0
    if 'l2_lambda' not in config.keys(): config['l2_lambda'] = 0

    train_model(writer, need_weight, loss_func, net, 
                train_loader, test_loader,
                optimizer, scheduler, 
                config['epochs'], args.model_path, 
                save_model=True, save_interval=1, 
                device=config['device'], model_info=config['model_info'],
                l1_lamba=config['l1_lambda'], l2_lamba=config['l2_lambda'])
    
    print('Training End at time: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    writer.close()