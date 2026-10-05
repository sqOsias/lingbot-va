# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME
import json

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    if not repo_list:
        raise FileNotFoundError(f"No LeRobot meta/info.json found under {config.dataset_path}")
    worker_count = min(num_init_worker, len(repo_list))
    if worker_count == 1:
        datasets_out_lst = [construct_func(repo_list[0])]
    else:
        with Pool(worker_count) as pool:
            datasets_out_lst = pool.map(construct_func, repo_list)
                
    return datasets_out_lst

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=8,
    ):
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        dataset_id = self.item_id_to_dataset_id[idx]
        cur_dset = self._datasets[dataset_id]
        local_idx = idx - self.acc_dset_num[dataset_id]
        sample = cur_dset[local_idx]
        if "episode_index" in sample:
            sample["dataset_numeric_id"] = torch.tensor(dataset_id,
                                                         dtype=torch.long)
        return sample

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        self.dataset_adapter = getattr(config, 'dataset_adapter', None)
        self.memory_alignment = None
        if self.dataset_adapter == 'rmbench':
            anchor_path = self.root / 'observation_anchors.json'
            with anchor_path.open() as stream:
                anchor_records = json.load(stream).values()
            self.rmbench_origins = {
                (int(record['episode_index']), int(record['start_frame'])):
                    np.asarray(record['origin'], dtype=np.float32)
                for record in anchor_records
            }
            alignment_path = getattr(config, "memory_alignment_path", None)
            if alignment_path:
                records = [json.loads(line) for line in
                           Path(alignment_path).read_text().splitlines()
                           if line.strip()]
                self.memory_alignment = {}
                for record in records:
                    key = (int(record["episode_index"]),
                           int(record["segment_start_frame"]),
                           int(record["segment_end_frame"]))
                    if key in self.memory_alignment:
                        raise ValueError(f"Duplicate memory alignment: {key}")
                    self.memory_alignment[key] = record
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        self.parse_meta()

    def parse_meta(self):
        out = []
        episode_indices = getattr(self.config, 'episode_indices', None)
        if episode_indices is not None:
            episode_indices = {int(value) for value in episode_indices}
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            if episode_indices is not None and int(episode_index) not in episode_indices:
                continue
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _rmbench_action_post_process(
        self, local_start_frame, local_end_frame, latent_frame_ids, action,
        episode_index,
    ):
        """Apply only RMBench's temporal, coordinate, and channel contract."""
        start, end = int(local_start_frame), int(local_end_frame)
        if int(latent_frame_ids[0]) != start:
            raise ValueError((start, int(latent_frame_ids[0])))
        frame_stride = int(latent_frame_ids[1] - latent_frame_ids[0])
        raw = np.asarray(action, dtype=np.float32)
        origin = self.rmbench_origins[(int(episode_index), start)]

        relative = []
        for offset in (0, 8):
            pose = np.concatenate(
                [origin[None, offset:offset + 7], raw[:, offset:offset + 7]],
                axis=0,
            )
            relative.extend([get_relative_pose(pose)[1:], raw[:, offset + 7:offset + 8]])
        raw = np.concatenate(relative, axis=1)

        # Adjacent windows overlap by one observation.  Rows [start, end-1)
        # therefore provide targets for physical frames [start+1, end-1].
        usable = end - start - 1
        group = frame_stride * 4
        raw = raw[:usable]

        if frame_stride != 4 or group != self.config.action_per_frame:
            raise ValueError(("RMBench APF16 contract", frame_stride, group,
                              self.config.action_per_frame))

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * group

        # Match LingBot's temporal contract for every independently sampled
        # window: latent/action frame 0 is an observation condition, so its
        # complete action group is dummy and must not contribute to the loss.
        # The parquet row at local ``start`` already stores physical action
        # ``start + 1``; it therefore begins at action frame 1 after padding.
        raw = np.pad(raw, ((group, 0), (0, 0)), constant_values=0)
        valid_rows = np.concatenate(
            [np.zeros(group, dtype=bool), np.ones(usable, dtype=bool)]
        )

        # The final RMBench window can contain more physical labels than fit
        # after reserving action frame 0. Follow the official implementation
        # and truncate at the window horizon before applying tail padding.
        raw = raw[:required_action_num]
        valid_rows = valid_rows[:required_action_num]
        tail_padding = required_action_num - len(raw)
        if tail_padding:
            raw = np.pad(raw, ((0, tail_padding), (0, 0)), constant_values=0)
            valid_rows = np.concatenate(
                [valid_rows, np.zeros(tail_padding, dtype=bool)]
            )

        if len(raw) != required_action_num:
            raise ValueError((start, end, len(raw), latent_frame_num, group))

        raw = np.pad(raw, ((0, 0), (0, 1)), constant_values=0)
        mask = np.broadcast_to(valid_rows[:, None], raw.shape).copy()
        mask[:, 16] = False
        raw = raw[:, self.config.inverse_used_action_channel_ids]
        mask = mask[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (raw - self.q01) / (self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = np.clip(action_aligned, -1.5, 1.5) * mask
        action_aligned = rearrange(
            action_aligned, "(f n) c -> c f n 1", f=latent_frame_num
        )
        mask = rearrange(mask, "(f n) c -> c f n 1", f=latent_frame_num)
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(mask).bool()

    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action,
                             episode_index=None):
        if self.dataset_adapter == 'rmbench':
            return self._rmbench_action_post_process(
                local_start_frame, local_end_frame, latent_frame_ids, action,
                episode_index,
            )
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = np.clip(action_aligned, -1.5, 1.5)
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(
            local_start_frame, local_end_frame, latent_frame_ids,
            ori_data_dict['action'], episode_index=episode_index,
        )

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        if self.memory_alignment is not None:
            key = (int(episode_index), int(local_start_frame),
                   int(local_end_frame))
            if key not in self.memory_alignment:
                raise KeyError(f"Missing memory alignment for {key}")
            record = self.memory_alignment[key]
            decisions = record["decisions"]
            if out_dict['latents'].shape[1] != int(record["latent_frames"]):
                raise ValueError((key, out_dict['latents'].shape,
                                  record["latent_frames"]))
            out_dict.update({
                "episode_index": torch.tensor(episode_index, dtype=torch.long),
                "segment_index": torch.tensor(record["segment_index"],
                                               dtype=torch.long),
                "global_frame_ids": torch.tensor(record["global_frame_ids"],
                                                  dtype=torch.long),
                "source_rgb_indices": torch.tensor(
                    record["latent_source_rgb_indices"], dtype=torch.long),
                "decision_frame_ids": torch.tensor(
                    [item["decision_global_frame_id"] for item in decisions],
                    dtype=torch.long),
                "target_frame_ids": torch.tensor(
                    [item["target_global_frame_id"] for item in decisions],
                    dtype=torch.long),
                "target_action_group_ids": torch.tensor(
                    [item["target_action_group_id"] for item in decisions],
                    dtype=torch.long),
                "video_target_mask": torch.tensor(
                    record["video_target_valid_by_latent"], dtype=torch.bool),
                "action_valid_slots_by_latent": torch.tensor(
                    record["action_valid_slots_by_latent"], dtype=torch.long),
                "reset_mask": torch.tensor(True, dtype=torch.bool),
            })
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
