import os
import re
import torch

ckpt_extension = ".tar"
ckpt_name_tmpl = "model.ckpt-{}" + ckpt_extension

def save(model_dir, epoch, net, optimizer, params=None):
    fname = ckpt_name_tmpl.format(epoch)
    fpath = os.path.join(model_dir, fname)

    ckpt_data = {
        "epoch": epoch,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "params": params or {}
    }

    torch.save(ckpt_data, fpath)

def load(model_dir, checkpoint):
    fname = ckpt_name_tmpl.format(checkpoint)
    fpath = os.path.join(model_dir, fname)

    ckpt_data = torch.load(fpath)

    epoch = ckpt_data.get("epoch")
    model_state = ckpt_data.get("model_state_dict")
    optimizer_state = ckpt_data.get("optimizer_state_dict")
    params = ckpt_data.get("params")

    return epoch, model_state, optimizer_state, params

def last(model_dir):
    regex = "^model.ckpt-(?P<checkpoint>[\d]+)\.tar" 
    
    ckpts_names = (f for f in os.listdir(model_dir) if f.endswith(ckpt_extension))
    ckpt_matches = (re.match(regex, f) for f in ckpts_names)
    ckpt_nums = [int(m.group("checkpoint")) for m in ckpt_matches if m]

    return max(ckpt_nums) if len(ckpt_nums) else 0