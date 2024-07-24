import torch


def trip_op(U, V, W, m, n):
    """
    triplets operator.
    Input:
        U, V, W: [batch, M, Nmodes]
        m, n: int
    Output:
        [batch, Nmodes]
    """
    assert U.shape[1] % 2 == 1, "M must be odd."
    assert U.shape[-1] == 1 or U.shape[-1] == 2, "Nmodes must be 1 or 2"

    p = U.shape[1] // 2
    if U.shape[-1] == 1:
        return U[:, p + m] * V[:, p + n] * W[:, p + m + n].conj()
    else:
        A = U[:, p + m] * W[:, p + m + n].conj()
        return (A + A.roll(1, dims=-1)) * V[:, p + n]


def triplets(E: torch.Tensor, m, n) -> torch.Tensor:
    '''
    Self triplets operator.
    Input:
        E: [batch, M, Nmodes]
        m: int or 1d  tensor.
        n: int or 1d  tensor.
    Output:
        [batch, Nmodes] or [batch, len(m), Nmodes]
    '''
    return trip_op(E, E, E, m, n)


def get_power(task_info, Nmodes, device):
    """
    task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]
    Nmodes: int
    device: torch.device
    -> torch.Tensor [batch]
    """
    P = (
        torch.tensor(1) if task_info == None else 10 ** (task_info[:, 0] / 10) / Nmodes
    )  # [batch] or ()
    P = P.to(device)
    return P
