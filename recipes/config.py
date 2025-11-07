"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
from omegaconf import OmegaConf


def save(filename, config):
    path = "/".join(filename.split("/")[:-1])
    if len(path) > 0:
        os.makedirs(path, exist_ok=True)

    with open(filename, "w") as f:
        OmegaConf.save(config=config, f=f.name)


class Config:
    def __init__(
        self,
        default_config = None,
    ):
        if default_config is None:
            default_config = {}

        # Initialize with default config
        self.config = OmegaConf.create(default_config)
        # Update config with command line arguments
        args = OmegaConf.from_cli()
        self.config = OmegaConf.merge(self.config, args)
           
    def get(self):
        return self.config
