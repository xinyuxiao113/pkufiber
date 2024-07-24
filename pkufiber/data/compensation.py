'''
Take CDC and DBP on dataset.

train.h5
    - group:
        - pulse
        - SymbTx  
        - SignalTx (optional)
        - SignalRx
        - Rx(sps=2,chid=0,method=frequency cut)
            - info            # [Pch, Fi, Rs, Nch]
            - Tx
            - Rx
            - Rx_CDC 
            - Rx_DBP%d
            - Rx_CDCDDLMS
            - Rx_DBP%dDDLMS
            - ...
        """
'''

import argparse, numpy as np, os, pickle, yaml, torch, time, h5py, jax
from functools import partial
from pkufiber.utils import calc_time
import pkufiber.dsp as dsp


def data_compensation(path: str, method: str, taps=32, lr=[1 / 2**6, 1 / 2**7], stps=1, rx_grp="Rx(sps=2,chid=0,method=frequency cut)", device='cuda:0'):  # type: ignore

    with h5py.File(path, "a") as f:
        for key in f.keys():
            group = f[key]
            assert isinstance(group, h5py.Group)
            subgrp = group[rx_grp]

            assert isinstance(subgrp, h5py.Group)
            Rx = torch.from_numpy(subgrp["Rx"][...]).to(torch.complex64)  # type: ignore
            Tx = torch.from_numpy(subgrp["Tx"][...]).to(torch.complex64)  # type: ignore
            Fs = torch.tensor([subgrp["Rx"].attrs["Fs(Hz)"]] * Rx.shape[0]).to(
                torch.float32
            )

            if method == "CDC":
                if "Rx_CDC" in subgrp.keys():
                    print(f"{rx_grp}/Rx_CDC already exists in {key}.")
                    E = torch.from_numpy(subgrp["Rx_CDC"][...]).to(torch.complex64)  # type: ignore
                else:
                    t0 = time.time()
                    E = dsp.cdc(Rx.to(device), Fs.to(device), group.attrs["distance(km)"] * 1e3)  # type: ignore  [B, Nfft, Nmodes]
                    t1 = time.time()
                    print(f"CDC time: {t1-t0}", flush=True)

                    data = subgrp.create_dataset("Rx_CDC", data=E.cpu().numpy())
                    data.dims[0].label = "batch"
                    data.dims[1].label = "time"
                    data.dims[2].label = "modes"
                    data.attrs.update(
                        {"sps": subgrp["Rx"].attrs["sps"], "start": 0, "stop": 0}
                    )

                if f"Rx_CDCDDLMS(taps={taps},lr={lr})" in subgrp.keys():
                    print(f"{rx_grp}/Rx_CDCDDLMS already exists in {key}.")
                else:
                    t0 = time.time()
                    sig_in, symb_in = jax.numpy.array(E.to("cpu")), jax.numpy.array(
                        Tx.to("cpu")
                    )
                    F = jax.vmap(dsp.mimoaf, in_axes=(0, 0, None, None, None, None))(
                        sig_in,
                        symb_in,
                        taps,
                        subgrp["Rx"].attrs["sps"],
                        2000,
                        tuple(lr),
                    )  # [B, Nfft, Nmodes]
                    F_data = torch.tensor(jax.device_get(F.val))
                    t1 = time.time()
                    print(f"DDLMS time: {t1-t0}", flush=True)

                    data = subgrp.create_dataset(
                        f"Rx_CDCDDLMS(taps={taps},lr={lr})", data=F_data
                    )
                    data.dims[0].label = "batch"
                    data.dims[1].label = "time"
                    data.dims[2].label = "modes"
                    data.attrs.update(
                        {
                            "sps": F.t.sps,
                            "start": F.t.start,
                            "stop": F.t.stop,
                            "lr": lr,
                            "taps": taps,
                        }
                    )

            elif method == "DBP":
                if f"Rx_DBP{stps}" in subgrp.keys():
                    print(f"{rx_grp}/Rx_DBP{stps} already exists in {key}.")
                    E = torch.from_numpy(subgrp[f"Rx_DBP{stps}"][...]).to(torch.complex64)  # type: ignore
                else:
                    t0 = time.time()
                    dz = group.attrs["Lspan(km)"] * 1e3 / stps  # type: ignore
                    power_dbm = torch.tensor([group.attrs["Pch(dBm)"]])
                    E = dsp.dbp(Rx.to(device), group.attrs["distance(km)"] * 1e3, dz, Fs.to(device), power_dbm.to(device))  # type: ignore  [B, Nfft, Nmodes]
                    t1 = time.time()
                    print(f"DBP time: {t1-t0}", flush=True)

                    data = subgrp.create_dataset(f"Rx_DBP{stps}", data=E.cpu().numpy())
                    data.dims[0].label = "batch"
                    data.dims[1].label = "time"
                    data.dims[2].label = "modes"
                    data.attrs.update(
                        {
                            "sps": subgrp["Rx"].attrs["sps"],
                            "start": 0,
                            "stop": 0,
                            "stps": stps,
                        }
                    )
                    data.asstr

                if f"Rx_DBP{stps}DDLMS(taps={taps},lr={lr})" in subgrp.keys():
                    print(f"{rx_grp}/Rx_DBP{stps}DDLMS already exists in {key}.")
                    continue
                else:
                    t0 = time.time()
                    sig_in, symb_in = jax.numpy.array(E.to("cpu")), jax.numpy.array(
                        Tx.to("cpu")
                    )
                    F = jax.vmap(dsp.mimoaf, in_axes=(0, 0, None, None, None, None))(
                        sig_in,
                        symb_in,
                        taps,
                        subgrp["Rx"].attrs["sps"],
                        2000,
                        tuple(lr),
                    )  # [B, Nfft, Nmodes]
                    F_data = torch.tensor(jax.device_get(F.val))
                    t1 = time.time()
                    print(f"DDLMS time: {t1-t0}", flush=True)

                    data = subgrp.create_dataset(
                        f"Rx_DBP{stps}DDLMS(taps={taps},lr={lr})", data=F_data
                    )
                    data.dims[0].label = "batch"
                    data.dims[1].label = "time"
                    data.dims[2].label = "modes"
                    data.attrs.update(
                        {
                            "sps": F.t.sps,
                            "start": F.t.start,
                            "stop": F.t.stop,
                            "lr": lr,
                            "taps": taps,
                        }
                    )


@calc_time
def main():
    parser = argparse.ArgumentParser(description="Simulation Configuration")
    parser.add_argument(
        "--path", type=str, default="dataset/test.h5", help="dataset path"
    )
    parser.add_argument(
        "--comp", type=str, default="CDC", help="method for compensation. CDC or DBP"
    )
    parser.add_argument(
        "--stps", type=int, default=1, help="steps per span for DBP. not used for CDC"
    )
    parser.add_argument(
        "--rx_grp",
        type=str,
        default="Rx(sps=2,chid=0,method=frequency cut)",
        help="Rx group name",
    )

    # mimoaf
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=[1 / 2**6, 1 / 2**7],
        help="DDLMS learning rate",
    )
    parser.add_argument("--taps", type=int, default=32, help="DDLMS taps")
    args = parser.parse_args()

    data_compensation(
        args.path,
        args.comp,
        taps=args.taps,
        lr=args.lr,
        stps=args.stps,
        rx_grp=args.rx_grp,
    )
