import os
import hydra
import numpy as np
import h5py
from scipy.interpolate import interp1d
from tqdm import tqdm
from hiss.utils import DATA_DIR


@hydra.main(config_path="conf/dataset", config_name="config", version_base=None)
def main(cfg):
    modalities = dict(cfg.freq).keys()
    data_path = f"{DATA_DIR}/{cfg.data_dir}"
    # Read file containing data
    with h5py.File(
        os.path.join(data_path, f"{cfg.data_dir}_{cfg.input_data_suffix}.h5")
    ) as hf:
        ts = {m: np.array(hf[f"{m}_timestamps"]) for m in modalities}
        data = {m: np.array(hf[m]) for m in modalities}
        eids = {m: np.array(hf[f"{m}_episode_ids"]) for m in modalities}
        num_eps = np.array([len(e) for _, e in eids.items()])
        assert np.all(num_eps == num_eps[0])
        num_episodes = num_eps[0] - 1

    HZ = cfg.freq
    new_eids = {m: [0] for m in modalities}
    new_data = {m: [] for m in modalities}
    new_ts = {m: [] for m in modalities}
    offset_id_list = {m: [] for m in modalities}
    durations = []
    ep_flag = [True] * num_episodes
    # Find initial and final indices for time alignment
    for n_ep in range(num_episodes):
        sids = {m: eids[m][n_ep] for m in modalities}
        endids = {m: eids[m][n_ep + 1] for m in modalities}
        dur_list = []
        for m in modalities:
            start_list = [ts[cm][sids[cm]] - ts[m][sids[m]] for cm in modalities]
            end_list = [ts[cm][endids[cm] - 1] for cm in modalities]
            s_off = np.amax(start_list)
            e_off = np.amin(end_list)
            offset_id = np.searchsorted(
                ts[m][sids[m] : endids[m]] - ts[m][sids[m]],
                s_off,
            )
            end_offset_id = (
                endids[m] - sids[m] - np.searchsorted(ts[m][sids[m] : endids[m]], e_off)
            )

            assert end_offset_id >= 0
            dur = ts[m][endids[m] - 1 - end_offset_id] - ts[m][sids[m] + offset_id]
            if dur < 0:
                ep_flag[n_ep] = False
                print(
                    f"Episode {n_ep}, modality {m}, duration {dur}, offset {offset_id}"
                )
            dur_list.append(dur)
            offset_id_list[m].append([offset_id, end_offset_id])
        durations.append(np.floor(min(dur_list)))

    # Convert data to regularly sampled at specified frequency
    for m in modalities:
        print(f"Processing modality {m}")
        for eid, (sid, lid) in tqdm(
            enumerate(zip(eids[m][:-1], eids[m][1:])), total=num_episodes
        ):
            if not ep_flag[eid]:
                continue
            sid_off = sid + offset_id_list[m][eid][0]
            lid_off = lid - offset_id_list[m][eid][1]
            # new_t = np.arange(0, ts[m][lid_off-1] - ts[m][sid_off], 1/HZ[m])
            if not "max_dur" in cfg:
                max_dur = durations[eid]
            else:
                max_dur = min(durations[eid], cfg.max_dur)
            new_t = np.arange(0, max_dur, 1 / HZ[m])
            _, time_ids = np.unique(
                ts[m][sid_off:lid_off] - ts[m][sid_off], return_index=True
            )
            interp_data = interp1d(
                ts[m][sid_off:lid_off][time_ids] - ts[m][sid_off],
                data[m][sid_off:lid_off][time_ids],
                axis=0,
                assume_sorted=True,
                fill_value="extrapolate",
            )(new_t)
            new_data[m].append(interp_data)
            new_eids[m].append(new_eids[m][-1] + len(interp_data))
            new_ts[m].append(new_t + ts[m][sid_off])
            # print(f"episode: {eid}, m: {len(interp_data)}")
        new_data[m] = np.concatenate(new_data[m], axis=0)
        print(new_data[m].shape, new_eids[m][-1])
    # print("before")
    # print([(np.mean(data[m]), np.std(data[m])) for m in modalities])
    # print("after")
    # print([(np.mean(new_data[m]), np.std(new_data[m])) for m in modalities])

    # Write data to new file
    with h5py.File(
        os.path.join(data_path, f"{cfg.data_dir}_{cfg.data_suffix}.h5"),
        "w",
    ) as hf:
        for m in modalities:
            hf.create_dataset(f"{m}_timestamps", data=np.concatenate(new_ts[m], axis=0))
            hf.create_dataset(f"{m}_episode_ids", data=new_eids[m])
            hf.create_dataset(f"{m}_frequency", data=HZ[m])
            hf.create_dataset(m, data=new_data[m])


if __name__ == "__main__":
    main()
