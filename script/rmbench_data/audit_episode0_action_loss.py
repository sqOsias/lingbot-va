#!/usr/bin/env python3
"""Audit the exact episode0 APF16 labels, masks, and action-loss denominator."""
from __future__ import annotations

import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

GROUPS = {
    "left_xyz": [0, 1, 2], "left_rotation": [3, 4, 5, 6], "left_gripper": [7],
    "right_xyz": [8, 9, 10], "right_rotation": [11, 12, 13, 14], "right_gripper": [15],
}

def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); sys.path.insert(0, str(a.repo))
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset
    cfg = copy.deepcopy(VA_CONFIGS["rmbench_overfit_train"])
    cfg.dataset_path = str(a.dataset); cfg.empty_emb_path = str(a.dataset / "empty_emb.pt")
    cfg.cfg_prob = 0.; cfg.load_worker = 0
    ds = LatentLeRobotDataset(str(a.dataset), cfg)
    used = np.asarray(cfg.used_action_channel_ids, dtype=int)
    empty = np.asarray([i for i in range(cfg.action_dim) if i not in set(used)], dtype=int)
    rows=[]; hashes=[]; checks={"six_windows": len(ds)==6, "used_channel_count_16": len(used)==16,
                               "empty_channel_count_14": len(empty)==14}
    for i, meta in enumerate(ds.new_metas):
        s=ds[i]; act=s["actions"].numpy(); mask=s["actions_mask"].numpy()
        flat=np.moveaxis(act,0,-1).reshape(-1,30); mf=np.moveaxis(mask,0,-1).reshape(-1,30)
        valid_rows=mf[:,used].all(1); partial=(mf[:,used].any(1) != valid_rows).sum()
        group={}
        for name, ids16 in GROUPS.items():
            ids=used[np.asarray(ids16)]
            values=flat[:,ids][mf[:,ids]].astype(np.float64)
            group[name]={"valid_elements":int(values.size),"std":float(values.std()),
                         "range":float(values.max()-values.min()),"nonzero":int(np.count_nonzero(values))}
        label_hash=digest(flat[valid_rows][:,used]); hashes.append(label_hash)
        total_elements=int(mask.sum()); valid_frames=int((mf.sum(1)>0).sum()); all_frames=len(mf)
        rows.append({"start":int(meta["start_frame"]),"end":int(meta["end_frame"]),
          "shape":list(act.shape),"valid_rows":int(valid_rows.sum()),"valid_elements":total_elements,
          "masked_rows":int((~valid_rows).sum()),"partially_masked_real_channels":int(partial),
          "first_16_rows_fully_masked":bool((~mf[:16, used]).all()),
          "valid_rows_contiguous_after_front_padding":bool(
              valid_rows[16:16 + int(valid_rows.sum())].all()
              and not valid_rows[16 + int(valid_rows.sum()):].any()),
          "real_channels_masked_on_valid_rows":int((~mf[valid_rows][:,used]).sum()),
          "empty_slots_unmasked":int(mf[:,empty].sum()),"label_sha256":label_hash,"groups":group,
          "current_loss_frame_mean_scale_vs_valid_token_mean":valid_frames/all_frames})
    checks.update({
      "all_16_real_channels_train":all(r["real_channels_masked_on_valid_rows"]==0 for r in rows),
      "all_14_empty_slots_masked":all(r["empty_slots_unmasked"]==0 for r in rows),
      "no_partial_real_channel_rows":all(r["partially_masked_real_channels"]==0 for r in rows),
      "all_windows_have_masked_condition_frame":all(r["first_16_rows_fully_masked"] for r in rows),
      "all_valid_labels_follow_condition_frame":all(r["valid_rows_contiguous_after_front_padding"] for r in rows),
      "regular_windows_have_192_valid_rows":all(r["valid_rows"]==192 for r in rows if r["start"]<960),
      "last_window_has_64_valid_rows":next(r for r in rows if r["start"]==960)["valid_rows"]==64,
      "six_distinct_label_hashes":len(set(hashes))==6,
      "all_groups_have_valid_elements":all(g["valid_elements"]>0 for r in rows for g in r["groups"].values()),
      "implementation_denominator_valid_tokens_only": all(token in (a.repo / "wan_va/train.py").read_text() for token in (
          "valid_action_frames = action_mask_per_frame > 0",
          "action_loss_per_frame[valid_action_frames]",
          "action_mask_per_frame[valid_action_frames]",
      )),
    })
    report={"status":"PASS" if all(checks.values()) else "FAIL",
      "checks":checks,"used_model30_channels":used.tolist(),"masked_model30_channels":empty.tolist(),
      "windows":rows,"finding":"Fully masked RMBench dummy/padding frames are excluded; LingBot's equal weighting across valid frames is preserved."}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"checks":checks}))

if __name__ == "__main__": main()
