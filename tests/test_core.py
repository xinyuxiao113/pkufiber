import torch
import pkufiber.core as core 


def test_TorchSignal():
    val = torch.randn(1, 10000, 2).to(torch.complex64)
    t = core.TorchTime(0, 0, 2)
    signal = core.TorchSignal(val=val, t=t)
    assert signal.val.shape == (1, 10000, 2)
    assert signal.t.start == 0
    assert signal.t.stop == 0
    assert signal.t.sps == 2
    assert signal.val.dtype == torch.complex64

