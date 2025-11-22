import torch


def test_optimize_scene_partitions_return_format():
    B, N = 1, 6
    sim = torch.rand(B, N, N)
    sim = 0.5 * (sim + sim.transpose(1, 2))
    res = __import__("vggt.models.aggregator", fromlist=["optimize_scene_partitions"]).optimize_scene_partitions(
        sim, num_groups=2, steps=10, lr=1e-1, lam=0.5, alpha=1.0, beta=1.0, patience=3, log_interval=5
    )
    assert isinstance(res, list)
    assert len(res) == B
    r0 = res[0]
    for k in [
        "soft_assignment",
        "hard_assignment",
        "counts",
        "group_indices",
        "group_indices_padded",
        "Lg",
        "best_loss",
        "steps",
        "elapsed_ms",
    ]:
        assert k in r0


def test_attention_token_merge_grouped():
    from vggt.layers.attention import Attention

    B = 1
    S = 2
    w = 4
    h = 4
    C = 32
    special = 5
    P = special + w * h
    N = S * P

    x = torch.randn(B, N, C)
    attn = Attention(dim=C, num_heads=4, patch_width=w, patch_height=h)
    y = attn(x, pos=None, global_merging=0)
    assert y.shape == (B, N, C)