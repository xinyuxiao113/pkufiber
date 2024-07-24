'''
Generate dataset.


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
        - Rx(sps=2,chid=0,method=filtering)
            - info
            - Tx
            - Rx
        ...
        """
'''

"""
Torch Simulation of optical fiber transmission.
"""
import argparse, numpy as np, os, pickle, yaml, torch, time
import h5py
from pkufiber.utils import calc_time
import pkufiber.simulation as sml


def set_seed():
    seed = int((time.time() * 1e6) % 1e9)  # 转换时间戳为整数种子
    np.random.seed(seed)
    tx_seed = np.random.randint(0, 2**32)
    ch_seed = np.random.randint(0, 2**32)
    rx_seed = np.random.randint(0, 2**32)
    return seed, tx_seed, ch_seed, rx_seed


def check_config(config):
    '''
    update: tx['sps'], ch['hz'], beta2
    '''
    beta2 = sml.channel.get_beta2(config["ch"]["D"], config["ch"]["Fc"])
    sps_min = sml.choose_sps(
        config["tx"]["Nch"], config["tx"]["freqspace"] * 1e9, config["tx"]["Rs"] * 1e9
    )
    hz_max = sml.choose_dz(
        config["tx"]["freqspace"] * 1e9,
        config["ch"]["Lspan"],
        config["tx"]["Pch_dBm"],
        config["tx"]["Nch"],
        beta2,
        config["ch"]["gamma"],
    )
    sps = max(config["tx"]["sps"], sps_min)
    hz = min(config["ch"]["hz"], hz_max)
    config["tx"]["sps"] = sps
    config["ch"]["hz"] = hz
    return config


def summary_config(tx_config, ch_config, rx_config):
    '''
    choose information to save.
    '''
    beta2 = sml.channel.get_beta2(ch_config["D"], ch_config["Fc"])
    grp_attrs = {
        "Nmodes": tx_config["Nmodes"],
        "Nch": tx_config["Nch"],
        "Rs(GHz)": int(tx_config["Rs"] / 1e9),
        "Pch(dBm)": tx_config["Pch_dBm"],
        "Lspan(km)": ch_config["Lspan"],
        "Fc(Hz)": ch_config["Fc"],
        "distance(km)": ch_config["Ltotal"],
        "beta2(s^2/km)": beta2,
        "gamma(/W/km)": ch_config["gamma"],
        "alpha(dB/km)": ch_config["alpha"],
        "D(ps/nm/km)": ch_config["D"],
        "Dpmd(ps/sqrt(km))": ch_config["Dpmd"],
        "NF(dB)": ch_config["NF"],
        "amp": ch_config["amp"],
        "PMD": ch_config["openPMD"],
        "Lcorr(km)": ch_config["Lcorr"],
        "M(QAM-order)": tx_config["M"],
        "batch": tx_config["batch"],
        "tx_sps": tx_config["sps"],
        "freqspace(Hz)": tx_config["freqspace"],
    }

    subgrp_attrs = {
        "sps": rx_config["rx_sps"],
        "chid": rx_config["chid"],
        "method": rx_config["method"],
        "seed": rx_config["seed"],
    }
    return grp_attrs, subgrp_attrs


def generate_names(config, seed):
    '''
    data structure names.
    '''
    grp_name = f'Nmodes{config["tx"]["Nmodes"]}_Rs{config["tx"]["Rs"]}_Nch{config["tx"]["Nch"]}_Pch{config["tx"]["Pch_dBm"]}_{seed}'
    subgrp_name = f'Rx(sps={config["rx"]["rx_sps"]},chid={config["rx"]["chid"]},method={config["rx"]["method"]})'
    return grp_name, subgrp_name


def create_data(file, data_name, data, attrs, dims_label):
    '''
    create data with dims label.
    '''
    data = file.create_dataset(data_name, data=data)
    for i, label in enumerate(dims_label):
        data.dims[i].label = label
    data.attrs.update(attrs)


def fiber_simulation(data_path, config):
    
    seed, tx_seed, ch_seed, rx_seed = set_seed()
    config = check_config(config)
    print(
        f'#######     Tx sps = {config["tx"]["sps"]},  simulation hz={config["ch"]["hz"]}km       #######',
        flush=True,
    )

    tx_data, tx_config = sml.wdm_transmitter(tx_seed, **config["tx"])
    trans_data, ch_config = sml.fiber_transmission(tx_data, ch_seed, **config["ch"])
    rx_data, rx_config = sml.wdm_receiver(trans_data, rx_seed, **config["rx"])
    grp_attrs, subgrp_attrs = summary_config(tx_config, ch_config, rx_config)
    grp_name, subgrp_name = generate_names(config, seed)

    with h5py.File(data_path, "a") as hdf:
        print(f"Creating Dataset: {grp_name}")
        grp = hdf.create_group(grp_name)
        subgrp = grp.create_group(subgrp_name)
        grp.attrs.update(grp_attrs)
        subgrp.attrs.update(subgrp_attrs)

        # - SignalTx
        if "save_SignalTx" in config.keys() and config["save_SignalTx"] == True:
            create_data(
                grp,
                "SignalTx",
                tx_data.signal.cpu().numpy(),
                {"sps": tx_config["sps"]},
                ["batch", "time", "modes"],
            )

        create_data(
            grp,
            "SymbTx",
            tx_data.symb.cpu().numpy(),
            {},
            ["batch", "time", "Nch", "Nmodes"],
        )

        create_data(
            grp,
            "pulse",
            tx_config["pulse"].cpu().numpy(),
            {"sps": tx_config["sps"]},
            ["time"],
        )

        create_data(
            grp,
            "SignalRx",
            trans_data.signal.cpu().numpy(),
            {"sps": tx_config["sps"]},
            ["batch", "time", "modes"],
        )

        create_data(
            subgrp,
            "Tx",
            rx_data.symb.squeeze(-2).cpu().numpy() / np.sqrt(10),
            {"sps": 1, "start": 0, "stop": 0, "Fs(Hz)": tx_config["Rs"]},
            ["batch", "time", "modes"],
        )

        create_data(
            subgrp,
            "Rx",
            rx_data.signal.cpu().numpy(),
            {
                "sps": rx_data.sps,
                "start": 0,
                "stop": 0,
                "Fs(Hz)": tx_config["Rs"] * rx_config["rx_sps"],
            },
            ["batch", "time", "modes"],
        )

        info = (
            torch.tensor(
                [
                    tx_config["Pch_dBm"],
                    tx_config["Fc"] + rx_config["chid"] * tx_config["freqspace"],
                    tx_config["Rs"] * tx_config["sps"],
                    tx_config["Nch"],
                ]
            )
            .repeat(rx_data.signal.shape[0], 1)
            .cpu()
            .numpy()
        )
        create_data(subgrp, "info", info, {}, ["batch", "task: Pch, Fi, Rs, Nch"])


@calc_time
def main():
    parser = argparse.ArgumentParser(description="Fiber Simulation")
    parser.add_argument(
        "--data_path", type=str, default="train.h5", help="path to save the data"
    )
    parser.add_argument(
        "--config_path", type=str, default="config.yaml", help="path to the config file"
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    fiber_simulation(args.data_path, config)
