import os, time, gc, json, pickle, argparse, math
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from data.util import *
import copy


def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def init_para_frompretrained(m, pm, share_para=False):
    m.wte.weight = pm.embed_tokens.weight
    m.wpe.weight = pm.embed_positions.weight

    for i in range(min(len(m.h), len(pm.layers))):
        m.h[i].ln_1.weight = pm.layers[i].fc1.weight if share_para else copy.copy(pm.layers[i].fc1.weight)
        m.h[i].ln_1.bias = pm.layers[i].fc1.bias if share_para else copy.copy(pm.layers[i].fc1.bias)
        # m.h[i].attn.c_attn.weight = pm.layers[i].attn.c_attn.weight if share_para else copy.copy(pm.layers[i].attn.c_attn.weight)
        # m.h[i].attn.c_attn.bias = pm.layers[i].attn.c_attn.bias if share_para else copy.copy(pm.layers[i].attn.c_attn.bias)
        # m.h[i].attn.c_proj.weight = pm.layers[i].attn.c_proj.weight if share_para else copy.copy(pm.layers[i].attn.c_proj.weight)
        # m.h[i].attn.c_proj.bias = pm.layers[i].attn.c_proj.bias if share_para else copy.copy(pm.layers[i].attn.c_proj.bias)
        m.h[i].attn = pm.layers[i].self_attn
        m.h[i].ln_2.weight = pm.layers[i].fc2.weight if share_para else copy.copy(pm.layers[i].fc2.weight)
        m.h[i].ln_2.bias = pm.layers[i].fc2.bias if share_para else copy.copy(pm.layers[i].fc2.bias)
        # m.h[i].mlp.c_fc.weight = pm.layers[i].mlp.c_fc.weight if share_para else copy.copy(pm.layers[i].mlp.c_fc.weight)
        # m.h[i].mlp.c_fc.bias = pm.layers[i].mlp.c_fc.bias if share_para else copy.copy(pm.layers[i].mlp.c_fc.bias)
        # m.h[i].mlp.c_proj.weight = pm.layers[i].mlp.c_proj.weight if share_para else copy.copy(pm.layers[i].mlp.c_proj.weight)
        # m.h[i].mlp.c_proj.bias = pm.layers[i].mlp.c_proj.bias if share_para else copy.copy(pm.layers[i].mlp.c_proj.bias)

    # m.ln_f.weight = pm.ln_f.weight if share_para else copy.copy(pm.ln_f.weight)
    # m.ln_f.bias = pm.ln_f.bias if share_para else copy.copy(pm.ln_f.bias)


def switch_schedule(schedule, mult, switch):
    """ Apply LR multiplier before iteration "switch" """

    def f(e):
        s = schedule(e)
        if e < switch:
            return s * mult
        return s

    return f


def linear_schedule(args):
    def f(e):
        if e <= args.warmup:
            return e / args.warmup
        return max((e - args.iterations) / (args.warmup - args.iterations), 0)

    return f
