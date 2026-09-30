"""Bounded training view over unchanged full-context RMBench latent files."""
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from .lerobot_latent_dataset import MultiLatentLeRobotDataset


class BoundedFullContextDataset(Dataset):
    def __init__(self, config, index_path):
        self.index_path = Path(index_path)
        self.records = [json.loads(line) for line in self.index_path.read_text().splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"Empty bounded index: {self.index_path}")
        self.full = MultiLatentLeRobotDataset(config=config)
        self.by_episode = defaultdict(list)
        for idx, record in enumerate(self.records):
            ep = int(record["episode_index"])
            if not 0 <= ep < len(self.full):
                raise ValueError((self.index_path, ep, len(self.full)))
            if int(record["origin_start_frame"]) != 0:
                raise ValueError((self.index_path, "origin", ep))
            self.by_episode[ep].append(idx)
        if sorted(self.by_episode) != list(range(len(self.full))):
            raise ValueError(f"Index does not include each source episode: {self.index_path}")
        self.window_counts = {ep: len(indices) for ep, indices in self.by_episode.items()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        episode = int(record["episode_index"])
        # The full-context source has exactly one action_config per episode.
        full = self.full[episode]
        start = int(record["global_latent_start"])
        end = int(record["global_latent_end"])
        target = int(record["target_latent_start"])
        frames = full["latents"].shape[1]
        if frames != int(record["episode_latent_frames"]) or not 0 <= start < target < end <= frames:
            raise ValueError((episode, start, target, end, frames))
        if full["actions"].shape[1] != frames or full["actions_mask"].shape[1] != frames:
            raise ValueError((episode, "video/action length mismatch"))
        output = {
            "latents": full["latents"][:, start:end].contiguous(),
            "actions": full["actions"][:, start:end].contiguous(),
            "actions_mask": full["actions_mask"][:, start:end].contiguous(),
            "text_emb": full["text_emb"],
            "global_latent_start": torch.tensor(start, dtype=torch.long),
            "history_frame_count": torch.tensor(target - start, dtype=torch.long),
            "episode_index": torch.tensor(episode, dtype=torch.long),
            "validation_weight": torch.tensor(1.0 / self.window_counts[episode]),
        }
        if not output["actions_mask"][:, target - start:].any():
            raise ValueError((episode, start, target, end, "no target actions"))
        return output


class EpisodeBalancedDistributedSampler(Sampler):
    """Sample episodes uniformly, then one window uniformly within each episode."""
    def __init__(self, dataset, num_replicas, rank, seed=42):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.episodes = sorted(dataset.by_episode)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + 1009 * self.epoch + self.rank)
        for step in range(self.num_samples):
            forced = os.environ.get("BLOCKS_SMOKE_FORCE_EPISODE")
            if forced is not None and self.epoch == 0 and self.rank == 1 and step == 1:
                episode = int(forced)
                choices = self.dataset.by_episode[episode]
                maximum = max(int(self.dataset.records[i]["global_latent_end"]) - int(self.dataset.records[i]["global_latent_start"]) for i in choices)
                choices = [i for i in choices if int(self.dataset.records[i]["global_latent_end"]) - int(self.dataset.records[i]["global_latent_start"]) == maximum]
            else:
                episode = self.episodes[torch.randint(len(self.episodes), (1,), generator=generator).item()]
                choices = self.dataset.by_episode[episode]
            yield choices[torch.randint(len(choices), (1,), generator=generator).item()]
