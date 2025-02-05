# %%
import logging
import os
import sys
import traceback

import torch

from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from omegaconf import OmegaConf

from training.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"


# %%
def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int):

    # CUDA runtime does not support `fork`
    torch.multiprocessing.set_start_method("spawn")

    single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=1)


def format_exception(e: Exception, limit=20):
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


# %%
def main(config_yaml_name) -> None:
    cfg = compose(config_name=config_yaml_name)
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", "experiment_log_dir"
        )
    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)

    # Priotrize cmd line args
    cfg.launcher.gpus_per_node = 1
    cfg.launcher.num_nodes = 1

    single_node_runner(cfg, 4500)


# %%
initialize_config_module("sam2", version_base="1.2")
register_omegaconf_resolvers()

# %%
training_config_path = (
    r"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\training\training_config.yaml"
)

# %%
main(training_config_path)

# %%
print("Finito")

# %%
