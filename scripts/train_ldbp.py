import pickle, torch, numpy as np, time, jax, matplotlib.pyplot as plt
import argparse, os , yaml, torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from torch.utils.tensorboard.writer import SummaryWriter

import pkufiber as pf
import pkufiber.dsp as dsp
import pkufiber.dsp.nonlinear_compensation.ldbp as ldbp
from pkufiber.data import FiberDataset, MixFiberDataset
from pkufiber.core import TorchInput, TorchSignal, TorchTime
from pkufiber.dsp.nonlinear_compensation.ldbp import FDBP, MetaDBP, downsamp
from pkufiber.dsp.nonlinear_compensation.loss import  mse, mse_rotation_free



def write_log(writer, epoch, train_loss, metric):
    '''
    writer: SummaryWriter
    epoch: int
    train_loss: float
    metric: dict
    '''
    writer.add_scalar('Loss/train',  train_loss, epoch)
    writer.add_scalar('Loss/test', metric['MSE'], epoch)
    writer.add_scalar('Metric/SNR', metric['SNR'], epoch)
    writer.add_scalar('Metric/BER', metric['BER'], epoch)
    writer.add_scalar('Metric/Qsq', metric['Qsq'], epoch)
    

def init_model(model_name, model_info):
    '''
    Load model.
        model_name: model name.
        model_info: model info.
    '''
    module = getattr(ldbp, model_name)
    dbp = module(**model_info)
    return dbp


def test_model(net, test_loader, taps=32, device='cuda:0', ber_discard=20000):
    '''
    Test DBP + ADF.
        net: LDBP module.
        device: cuda device.
        taps: ADF filter length.
        power: power of the signal.
        Nch: number of channels.
        Rs: symbol rate.
        test_path: path to test data.
        ber_discard: discard the first ber_discard samples.
        Nsymb: number of symbols to test.

    Return:
        {'MSE': mse, 'BER': ber, 'Qsq': Qsq(ber), 'Q_path': Q_path}, (z1, z2)
    '''

    # load data
    assert test_loader.batch_size == 1, 'Batch size should be 1'

    z1_list = []
    z2_list = []
    for Rx, Tx, info in test_loader:
        signal = TorchSignal(val=Rx, t=TorchTime(0,0,2)).to(device)
        symb = TorchSignal(val=Tx, t=TorchTime(0,0,1)).to(device)
        info = info.to(device)

        # DBP
        with torch.no_grad():
            y = net(signal, info)
    
        # ADF
        sig_in = jax.numpy.array(y.val[0].cpu().numpy())
        symb_in = jax.numpy.array(symb.val[0, y.t.start//y.t.sps:y.t.stop//y.t.sps].cpu().numpy())
        z = dsp.mimoaf(sig_in, symb_in, taps=taps)      

        # metric
        z1 = torch.tensor(jax.device_get(z.val))
        z2 = torch.tensor(jax.device_get(symb_in[z.t.start:z.t.stop]))

        assert z1.shape[0] > ber_discard, 'ber_discard is too large'
        z1_list.append(z1[ber_discard:])
        z2_list.append(z2[ber_discard:])
    
    z1 = torch.cat(z1_list, dim=0)
    z2 = torch.cat(z2_list, dim=0)
    
    mse_value = mse(z1, z2)
    power_value = torch.mean(torch.abs(z2)**2).item()
    ber = np.mean(pf.ber(z1, z2)['BER'])
    q_path = pf.qfactor_path(z1, z2, Ntest=10000, stride=1000)
    snr = 10 * np.log10(power_value / mse_value)
    

    return {'MSE': mse_value, 'SNR': snr, 'BER': ber, 'Qsq': pf.qfactor(ber), 'Qsq_path': q_path}, (z1, z2)



def train_model(writer, net: nn.Module, conv: nn.Module, train_loader, test_loader, optimizer, scheduler , epochs:int, model_path: str, save_model=True, save_interval=1, device='cuda:0', model_info={}):
    '''
    Train DBP + ADF.
        writer: SummaryWriter.
        net: LDBP module.
        conv: ADF module.
        train_loader: DataLoader for training.
        optimizer: optimizer.
        scheduler: scheduler.
        epochs: number of epochs.
        model_path: path to save models.  
        save_log: save logs or not.
        save_model: save model or not.
        save_interval: save model every save_interval epochs.
        device: cuda device.
        model_info: model info.
    '''

    # setting
    loss_fn = mse_rotation_free  # MSE

    metric, (rx, tx) = test_model(net, test_loader, taps=32, device=device)
    print('Test BER: %.5f, Qsq: %.5f, MSE: %.5f' % (metric['BER'], metric['Qsq'], metric['MSE']), flush=True)
    write_log(writer, 0, 0, metric)

    for epoch in range(0, epochs + 1): 
        N = len(train_loader)
        train_loss = 0
        t0 = time.time()
        for i,(Rx, Tx, info) in enumerate(train_loader):
            signal_input = TorchSignal(val=Rx, t=TorchTime(0,0,2)).to(device)
            signal_output = TorchSignal(val=Tx, t=TorchTime(0,0,1)).to(device)
            info = info.to(device)

            y = net(signal_input, info)  # [B, L, N]
            y = conv(y)
            truth = signal_output.val[:, y.t.start:y.t.stop]     # [B, L, N]
            loss = loss_fn(y.val, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            writer.add_scalar('Loss/train_batch', loss.item(), epoch*N+i)
        t1 = time.time()
        scheduler.step()

        metric, (rx, tx) = test_model(net, test_loader, taps=32, device=device)

        write_log(writer, epoch, train_loss/len(train_loader), metric)
        
        # plot res['Qsq_path'] in tensorboard, Qsq_path is a list of Qsq values
        fig = plt.figure(figsize=(5,5))
        plt.plot(metric['Qsq_path'])
        plt.grid()
        writer.add_figure('Qsq_path', fig, epoch)

        print('Epoch: %d, Loss: %.5f, time: %.5f' % (epoch, train_loss/N, t1-t0), flush=True)
        print('Test BER: %.5f, Qsq: %.5f, MSE: %.5f' % (metric['BER'], metric['Qsq'], metric['MSE']), flush=True)

        if epoch % save_interval == 0 and save_model:
            ckpt = {
                'dbp_info': model_info,
                'dbp_param': net.state_dict(),
                'conv_param': conv.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(ckpt, model_path + f'/{epoch}.pth')
            print('Model saved')




def main():
    parser = argparse.ArgumentParser(description='Train DBP Model with config.')
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')
    parser.add_argument('--log_path', type=str, default='log', help='path to save logs')
    parser.add_argument('--model_path', type=str, default='model', help='path to save models')
    args = parser.parse_args()
    with open(args.config, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    writer = SummaryWriter(args.log_path)
    for k,v in config.items(): writer.add_text(k, str(v))
    print('*****  Experitment: %s *****' % args.log_path)
    print('Training Start at time: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # load model
    dbp = init_model(config['dbp_name'], config['dbp_info'])
    conv = downsamp(taps=64, Nmodes=config['dbp_info']['Nmodes'], sps=2, init='zeros').to(config['device'])
    dbp = dbp.to(config['device'])
    conv = conv.to(config['device'])

    if config['dbp_name'] == 'FDBP_trainD':
        if config['pretrainD'] == True:  
            dbp.linear.train_filter(lr=3e-4, epoch=6000)
        # fix dbp.linear
        if config['trainD_again'] == False:
            for param in dbp.linear.parameters():
                param.requires_grad = False


    # optimizer
    optimizer = torch.optim.Adam([{'params': dbp.parameters(), 'lr': config['dbp_lr']}, {'params': conv.parameters(), 'lr': config['conv_lr']}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_step'], gamma=config['decay_gamma'])
    
    # load data
    config['train_data']['window_size'] = config['train_data']['strides'] +  dbp.overlaps + conv.overlaps
    config['test_data']['window_size'] = config['test_data']['strides'] +  dbp.overlaps + conv.overlaps
    train_data = MixFiberDataset(**config['train_data'])
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=False)
    test_data = MixFiberDataset(**config['test_data'])
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)
 
    # train
    model_path = args.model_path
    
    train_model(writer, dbp, conv, 
                train_loader, test_loader, 
                optimizer, scheduler, config['epochs'],
                model_path, 
                device=config['device'],
                model_info=config['dbp_info'])
    
    print('Training Finished at time ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    writer.close()