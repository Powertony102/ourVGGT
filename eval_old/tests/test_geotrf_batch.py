import os
import sys
import torch
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE, "CUT3R", "src"))

from dust3r.utils.geometry import geotrf, inv


def make_inputs(B=1, H=8, W=8):
    cam = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    pts = torch.randn(B, H, W, 3, dtype=torch.float32)
    return cam, pts


def test_geotrf_batch_match():
    cam, pts = make_inputs(2, 4, 5)
    out = geotrf(inv(cam), pts)
    assert out.shape == pts.shape


def test_geotrf_mismatch_fix():
    cam, pts = make_inputs(1, 3, 3)
    pts_np = pts[0].cpu().numpy()
    pts_fix = torch.from_numpy(pts_np.astype(np.float32)).unsqueeze(0)
    out = geotrf(inv(cam), pts_fix)
    assert out.shape == pts_fix.shape


if __name__ == "__main__":
    test_geotrf_batch_match()
    test_geotrf_mismatch_fix()
    print("ok")