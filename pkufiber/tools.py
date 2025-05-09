
import pkufiber as pf 
import torch 
import numpy as np 
import jax
import matplotlib.pyplot as plt 
import seaborn as sns
from torch.utils.data import DataLoader
import seaborn


def test_model(net, test_loader, device, kernel_size=32):
    """
    测试模型性能
    
    Args:
        net: 神经网络模型
        test_loader: 测试数据加载器
        device: 计算设备(cuda/cpu)
        kernel_size: 卷积核大小
        
    Returns:
        Q因子值
    """
    for Rx, Tx,info in test_loader:
        Rx = Rx.transpose(1,2).to(torch.complex64).to(device)
        Tx = Tx.to(device)
        break

    t = kernel_size // 4
    with torch.no_grad():
        y = net(Rx).transpose(1,2)
        y0 = Tx[:,t:-(t-1),:]

    return pf.qfactor_all(y, y0)


def train_static_filter(kernel_size, train_loader, test_loader, epochs=100):
    """
    训练静态滤波器模型
    
    Args:
        kernel_size: 卷积核大小
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        
    Returns:
        net: 训练好的模型
        Q_list: 训练过程中的Q因子列表
    """
    from pkufiber.dsp.layers import ComplexConv1d, ComplexLinear
    net = ComplexConv1d(2, 2, kernel_size, stride=2)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    Q_list = []

    for epoch in range(epochs):
        train_loss = 0
        for Rx, Tx,info in train_loader:
            Rx = Rx.transpose(1,2).to(torch.complex64).to(device)
            Tx = Tx.to(device)        
            t = kernel_size // 4    
            y = net(Rx).transpose(1,2)
            y0 = Tx[:,t:-(t-1),:]
            loss = torch.mean(torch.abs(y - y0)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()

        Q_test = test_model(net, test_loader, device, kernel_size=kernel_size)
        Q_list.append(Q_test)

        if epoch % 5 == 0:
            print('Epoch:', epoch, 'Loss:', train_loss/len(train_loader), 'Q:', Q_test)
        scheduler.step()
    
    return net, Q_list


def show_q_path(Qs, labels, title='ADF'):
    """
    绘制Q因子随符号索引的变化曲线
    
    Args:
        Qs: Q因子数据列表
        labels: 每条曲线的标签
        title: 图表标题
    """
    seaborn.set_style('whitegrid', {
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5
    })

    styles = ['-*', '-+', '-o', '-o', '-v', '-p']
    plt.figure(dpi=200)
    
    for i in range(len(Qs)):
        style = styles[i % len(styles)]
        plt.plot(Qs[i], style, label=labels[i], markersize=6, linewidth=2)
        print(labels[i], np.mean(Qs[i][5:]))

    plt.legend()
    plt.xlabel('Symbol Index (/1e4)', fontsize=14)
    plt.ylabel('Q Factor (dB)', fontsize=14)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.title(title, fontsize=16)
    plt.rcParams.update({
    'font.family': 'Arial',  # 使用 Arial 字体
    'font.size': 10,         # 合适的字体大小
    'axes.linewidth': 1,     # 适当的线条粗细
    'axes.grid': True,       # 显示网格线
    'grid.alpha': 0.3        # 网格线透明度
   })


def show_xpm_noise(Rx, Tx, theta, alg='ADF', window=20000, pol_mode=0, smooth_taps=10):
    """
    显示XPM相位噪声和算法跟踪结果
    
    Args:
        Rx: 接收信号
        Tx: 发送信号
        theta: 算法跟踪的相位
        alg: 算法名称
        window: 显示窗口大小
        pol_mode: 偏振模式
        smooth_taps: 平滑窗口大小
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(8, 4), dpi=200)

    real_phase = - np.angle(Rx[::2].numpy() / Tx.numpy())
    real_phase = np.convolve(real_phase[:, pol_mode], np.ones(smooth_taps)/smooth_taps, mode='same')

    plt.plot(real_phase[0:window], label='XPM Phase Noise', linewidth=1, color='#1f77b4')
    plt.plot(theta[0:window, pol_mode], label=f'Phase Tracked by {alg} Algorithm', linewidth=1, linestyle='--', color='#ff7f0e')

    plt.title('ADF Tracking XPM Phase Noise', fontsize=16, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Phase (radians)', fontsize=14)
    plt.legend(loc='upper right', fontsize=12, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(np.mean(real_phase) - 0.6, np.mean(real_phase) + 0.6)

    plt.rcParams.update({
       'font.family': 'Arial',  # 使用 Arial 字体
       'font.size': 10,         # 合适的字体大小
       'axes.linewidth': 1,     # 适当的线条粗细
       'axes.grid': True,       # 显示网格线
       'grid.alpha': 0.3        # 网格线透明度
   })


def plot_convolution_kernels(kernel, label, title):
    """
    绘制卷积核权重
    
    Args:
        kernel: 卷积核权重
        label: 图例标签
        title: 图表标题
    """
    sns.set_style('darkgrid')
    plt.figure(figsize=(10, 6), dpi=200)

    plt.plot(kernel, label=label, linewidth=6, linestyle='-', marker='o', markersize=5)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('taps m', fontsize=14)
    plt.ylabel('|w[m]|', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.show()
