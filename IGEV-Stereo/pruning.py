import sys
import argparse
import torch
import torch.nn as nn
import torch_pruning as tp

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1].as_posix()
sys.path.append(ROOT_DIR)

def prune(model, amount, img_size=(320, 736)):
    h, w = img_size

    # Importance criteria
    example_inputs = [torch.randn(1, 3, h, w).cuda(), torch.randn(1, 3, h, w).cuda()]
    imp = tp.importance.MagnitudeImportance(p=2)

    # Ignore some layers, e.g., the output layer
    ignored_layers = []

    for name, layer in model.named_modules():
        # print(name)
        if (
            False 
            # or "cnet" in name
            or "update_block" in name
            # or "context_zqr_convs" in name
            # or "feature" in name
            # or "stem" in name
            # or "spx" in name
            # or "conv" == name
            or "desc" in name
            # or "corr_stem" in name
            # or "corr_feature_att" in name
            or "cost_agg" in name
            or "classifier" in name
            # or "cnet.layer1" in name
            # or "cnet.layer2.0.conv2" in name
            # or "cnet.layer2.1" in name
            # or "cnet.layer3" in name
            # or "cnet.layer4" in name
            # or "cnet.layer4.1.conv2" in name
            # or "cnet.layer5" in name
            # or "cnet.outputs04" in name
            # or "cnet.outputs08" in name
            # or "cnet.outputs16" in name
        ):
            ignored_layers.append(layer)
    # print(ignored_layers)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=amount,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
        global_pruning=True
    )

    pruner.step()
