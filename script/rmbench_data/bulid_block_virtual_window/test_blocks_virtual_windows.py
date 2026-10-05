#!/usr/bin/env python3
"""Fast index-contract checks; run before any GPU smoke test."""
from script.rmbench_data.build_blocks_virtual_windows import windows_for_episode


def check(frames, maximum, history):
    records = windows_for_episode(20, frames, maximum, history)
    coverage = [0] * frames
    for record in records:
        start = record['global_latent_start']
        end = record['global_latent_end']
        target = record['target_latent_start']
        assert 0 <= start < target < end <= frames, record
        assert end - start <= maximum, record
        assert target - start <= history, record
        for frame in range(target, end):
            coverage[frame] += 1
    assert coverage == [0] + [1] * (frames - 1), (frames, coverage)


if __name__ == '__main__':
    for frames in (2, 33, 34, 64, 65, 152, 256):
        check(frames, 64, 32)
    for frames in (2, 49, 50, 152):
        check(frames, 48, 24)
    print('PASS: bounded-window target coverage')
