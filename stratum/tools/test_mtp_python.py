#!/usr/bin/env python3
"""
Reference MTP forward in Python — used to validate the C implementation.

Loads the BF16 model weights, runs the main forward through the prompt, then
runs MTP on (sampled_token, main_hidden) and reports MTP top-K predictions.
"""

import argparse
import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as F


def load_safetensors(path):
    raw = Path(path).read_bytes()
    hdr_len = struct.unpack('<Q', raw[:8])[0]
    hdr = json.loads(raw[8:8 + hdr_len])
    body_off = 8 + hdr_len
    return hdr, raw[body_off:]


def get_tensor(hdr, body, key, dtype=torch.bfloat16):
    m = hdr[key]
    off, end = m['data_offsets']
    raw = body[off:end]
    dt_map = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}
    dt = dt_map[m['dtype']]
    t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(m['shape'])
    return t.float()


def rms_norm(x, weight, eps=1e-6):
    """Qwen3.5 RMSNorm: y = (x / sqrt(mean(x^2)+eps)) * (1 + weight)"""
    var = x.pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(var + eps)
    return y * (1.0 + weight)


def mtp_forward(prev_token_id, main_hidden, hdr, body, embed_w, position):
    """Run MTP forward. Return logits over vocab."""
    e = embed_w[prev_token_id].float()
    h = main_hidden.float()

    pre_e = get_tensor(hdr, body, 'mtp.pre_fc_norm_embedding.weight')
    pre_h = get_tensor(hdr, body, 'mtp.pre_fc_norm_hidden.weight')
    e_n = rms_norm(e, pre_e)
    h_n = rms_norm(h, pre_h)
    print(f'[py_trace] e[0..7]    = {" ".join(f"{v:.6f}" for v in e[:8].tolist())}')
    print(f'[py_trace] e_n[0..7]  = {" ".join(f"{v:.6f}" for v in e_n[:8].tolist())}')
    print(f'[py_trace] h[0..7]    = {" ".join(f"{v:.6f}" for v in h[:8].tolist())}')
    print(f'[py_trace] h_n[0..7]  = {" ".join(f"{v:.6f}" for v in h_n[:8].tolist())}')

    fc_w = get_tensor(hdr, body, 'mtp.fc.weight')
    fc_in = torch.cat([e_n, h_n])
    x = fc_w @ fc_in
    print(f'[py_trace] after_fc: x[0..7] = {" ".join(f"{v:.6f}" for v in x[:8].tolist())}')

    in_ln_w = get_tensor(hdr, body, 'mtp.layers.0.input_layernorm.weight')
    post_ln_w = get_tensor(hdr, body, 'mtp.layers.0.post_attention_layernorm.weight')
    q_norm_w = get_tensor(hdr, body, 'mtp.layers.0.self_attn.q_norm.weight')
    k_norm_w = get_tensor(hdr, body, 'mtp.layers.0.self_attn.k_norm.weight')
    final_norm = get_tensor(hdr, body, 'mtp.norm.weight')

    q_w = get_tensor(hdr, body, 'mtp.layers.0.self_attn.q_proj.weight')
    k_w = get_tensor(hdr, body, 'mtp.layers.0.self_attn.k_proj.weight')
    v_w = get_tensor(hdr, body, 'mtp.layers.0.self_attn.v_proj.weight')
    o_w = get_tensor(hdr, body, 'mtp.layers.0.self_attn.o_proj.weight')

    gate_w = get_tensor(hdr, body, 'mtp.layers.0.mlp.gate_proj.weight')
    up_w = get_tensor(hdr, body, 'mtp.layers.0.mlp.up_proj.weight')
    down_w = get_tensor(hdr, body, 'mtp.layers.0.mlp.down_proj.weight')

    residual = x.clone()
    xn = rms_norm(x, in_ln_w)

    H_DIM = 1024
    HEAD = 256
    NQ = 8
    NK = 2
    GROUP = NQ // NK
    ROT_DIM = 64
    THETA = 1e7
    EPS = 1e-6

    q_full = q_w @ xn
    k = k_w @ xn
    v = v_w @ xn

    q_full = q_full.view(NQ, HEAD * 2)
    q = q_full[:, :HEAD].clone()
    gate = q_full[:, HEAD:].clone()
    k = k.view(NK, HEAD)
    v = v.view(NK, HEAD)

    q = rms_norm(q, q_norm_w)
    k = rms_norm(k, k_norm_w)

    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROT_DIM, 2).float() / ROT_DIM))
    angles = position * inv_freq
    cos = angles.cos()
    sin = angles.sin()
    def rope(t):
        t_rot = t[..., :ROT_DIM].clone()
        x_e = t_rot[..., 0::2]
        x_o = t_rot[..., 1::2]
        nx_e = x_e * cos - x_o * sin
        nx_o = x_e * sin + x_o * cos
        out = t.clone()
        out[..., 0:ROT_DIM:2] = nx_e
        out[..., 1:ROT_DIM:2] = nx_o
        return out
    q = rope(q)
    k = rope(k)

    scale = 1.0 / (HEAD ** 0.5)
    head_out = torch.zeros(NQ, HEAD)
    for h in range(NQ):
        kv = h // GROUP
        logit = scale * (q[h] * k[kv]).sum()
        head_out[h] = v[kv]
    head_out = head_out.flatten()
    head_out = head_out * torch.sigmoid(gate.flatten())

    attn_out = o_w @ head_out
    x = residual + attn_out

    residual2 = x.clone()
    xn2 = rms_norm(x, post_ln_w)
    g = gate_w @ xn2
    u = up_w @ xn2
    mlp_acc = F.silu(g) * u
    mlp_out = down_w @ mlp_acc
    x = residual2 + mlp_out

    final = rms_norm(x, final_norm)
    logits = embed_w.float() @ final
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='../model/model.safetensors-00001-of-00001.safetensors')
    ap.add_argument('--main-hidden', required=True,
                    help='space-separated 1024 floats representing main g_xn')
    ap.add_argument('--prev-token', type=int, required=True)
    ap.add_argument('--position', type=int, required=True)
    ap.add_argument('--top-k', type=int, default=10)
    args = ap.parse_args()

    hdr, body = load_safetensors(args.model)
    embed_w = get_tensor(hdr, body, 'model.language_model.embed_tokens.weight')

    main_h = torch.tensor([float(x) for x in args.main_hidden.split()])
    assert main_h.numel() == 1024, f'expected 1024 floats got {main_h.numel()}'

    logits = mtp_forward(args.prev_token, main_h, hdr, body, embed_w, args.position)
    top = torch.topk(logits, args.top_k)
    for i in range(args.top_k):
        print(f'  rank {i}: id={int(top.indices[i])} logit={float(top.values[i]):.4f}')


if __name__ == '__main__':
    main()
