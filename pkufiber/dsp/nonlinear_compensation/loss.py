import torch, numpy as np
import scipy.constants as const, scipy.special as special


def mse(x, y, epoch=0):
    return torch.mean(torch.abs(x - y) ** 2)

def p_mse(x, y, epoch=0, p=3.0):
    return torch.mean(torch.abs(x - y) ** p)


def weight_mse(x, y, epoch=0, p=3.0, weight=1.0):
    return torch.mean(torch.abs(x - y) ** 2 * weight**(p-2))


def avg_phase(x, y):
    return torch.angle(torch.mean(x * torch.conj(y), dim=1, keepdim=True))

def mse_rotation_free(x, y):
    # x, y: [batch, L, Nmodes]
    theta = avg_phase(x, y)
    return torch.mean(torch.abs(torch.exp(-1j * theta) * x - y) ** 2)

def well(x, mu: float=4):
    return torch.sigmoid(mu*(x - 0.3162)) + torch.sigmoid(mu*(-x - 0.3162))

def adaptive_ber(predict, truth, epoch=0):
    mu = 4*1.1**(epoch/2)
    error = predict - truth
    dis = torch.max(torch.abs(error.real), torch.abs(error.imag))
    return torch.mean(well(dis, mu))


def adaptive_ber_v2(predict, truth, epoch=0):
    mu = 8*1.1**epoch
    error = predict - truth
    dis = torch.max(torch.abs(error.real), torch.abs(error.imag))
    return torch.mean(well(dis, mu))


def adaptive_ber_v3(predict, truth, epoch=0):
    mu = 16*1.1**epoch
    error = predict - truth
    dis = torch.max(torch.abs(error.real), torch.abs(error.imag))
    return torch.mean(well(dis, mu))

def adaptive_ber_v4(predict, truth, epoch=0):
    mu = 8*1.1**min(epoch, 10)
    error = predict - truth
    dis = torch.max(torch.abs(error.real), torch.abs(error.imag))
    return torch.mean(well(dis, mu))

def adaptive_ber_v5(predict, truth, epoch=0):
    mu = 8*1.1**10
    error = predict - truth
    dis = torch.max(torch.abs(error.real), torch.abs(error.imag))
    return torch.mean(well(dis, mu))

def adaptive_ber_v6(predict, truth, epoch=0):
    mu = 8*1.1**min(epoch+10, 20)
    error = predict - truth
    dis = torch.max(torch.abs(error.real), torch.abs(error.imag))
    return torch.mean(well(dis, mu))



def l1_l2_regularization_loss(model, l1_lambda=1e-5, l2_lambda=1e-4) -> torch.Tensor:
    l1_loss = 0.0
    l2_loss = 0.0
    
    for param in model.parameters():
        if param.requires_grad:
            l1_loss += torch.sum(torch.abs(param))
            l2_loss += torch.sum(param ** 2)
    
    total_loss = l1_loss * l1_lambda  +  l2_loss * l2_lambda
    return total_loss # type: ignore
