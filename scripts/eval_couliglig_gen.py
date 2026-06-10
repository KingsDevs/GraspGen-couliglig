# Standalone generator quality eval on the held-out valid split.
# Reuses the trained diffusion model + valid dataloader, then for each object
# reports precision/recall at several match radii (to disambiguate "tight
# threshold" from "genuinely bad grasps"), the nearest-GT position/rotation
# error distribution, and confidence/likelihood ranges. Dumps predicted + GT
# grasps to .npz per object so they can be inspected in meshcat separately.
import os
import numpy as np
import hydra
import torch
from omegaconf import DictConfig
from scipy.spatial import cKDTree

from grasp_gen.utils.train_utils import get_data_loader, to_gpu, to_cpu
from grasp_gen.models.grasp_gen import GraspGenGenerator


def nearest_stats(pred, gt):
    """pred,gt: [N,4,4]. For each GT, distance to nearest pred (position, m)
    and the rotation geodesic (deg) to that nearest-position pred."""
    pp, gp = pred[:, :3, 3], gt[:, :3, 3]
    tree = cKDTree(pp)
    d, idx = tree.query(gp, k=1)  # nearest pred for each gt
    # rotation geodesic to the matched pred
    Rg, Rp = gt[:, :3, :3], pred[idx, :3, :3]
    Rrel = np.einsum("nij,nkj->nik", Rg, Rp)  # Rg @ Rp^T
    tr = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    rot_deg = np.degrees(np.arccos(tr))
    return d, rot_deg


def coverage(a, b, r):
    """fraction of poses in `a` that have some pose in `b` within r (position)."""
    tree = cKDTree(b[:, :3, 3])
    hit = [len(tree.query_ball_point(p, r)) > 0 for p in a[:, :3, 3]]
    return float(np.mean(hit))


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = cfg.eval.output_dir or "."
    os.makedirs(out_dir, exist_ok=True)

    _, loader = get_data_loader(
        cfg.eval, cfg.data, cfg.eval.split, None,
        use_ddp=False, training=False, inference=True,
    )
    model = GraspGenGenerator.from_config(cfg.diffusion)
    ckpt = torch.load(cfg.eval.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.cuda().eval()

    radii = [0.02, 0.05, 0.10, 0.20]
    print("\n" + "=" * 100)
    print(f"{'object':<20}{'#pred':>6}{'#gt':>6}{'medDist(m)':>11}"
          f"{'medRot(deg)':>12}" + "".join(f"recall@{r}".rjust(11) for r in radii))
    print("=" * 100)

    for data in loader:
        if data is None:
            continue
        to_gpu(data)
        with torch.no_grad():
            outputs, _, stats = model.infer(data, return_metrics=True)
        to_cpu(data); to_cpu(outputs)

        for j in range(len(data["scene"])):
            name = data["scene"][j].split("/")[-1]
            pred = np.asarray(outputs["grasps_pred"][j])
            gt = np.asarray(data["grasps_highres"][j])
            if pred.ndim != 3 or gt.ndim != 3 or len(pred) == 0 or len(gt) == 0:
                print(f"{name:<20} (skipped: pred {pred.shape} gt {gt.shape})")
                continue
            d, rot = nearest_stats(pred, gt)
            recalls = [coverage(gt, pred, r) for r in radii]
            lik = np.asarray(outputs["likelihood"][j]).reshape(-1)
            print(f"{name:<20}{len(pred):>6}{len(gt):>6}{np.median(d):>11.4f}"
                  f"{np.median(rot):>12.1f}"
                  + "".join(f"{r:>11.3f}" for r in recalls))
            np.savez(os.path.join(out_dir, f"{name}.npz"),
                     pred=pred, gt=gt, likelihood=lik)
    print("=" * 100)
    print(f"npz dumps (pred+gt grasps) written to {out_dir}")


if __name__ == "__main__":
    main()
