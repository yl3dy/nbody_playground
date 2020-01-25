"""
Example dummy engine.

Just writes the initial body config all over.

"""
from ..common import  SystemState, GlobalConfig
from .. import common
import progressbar

METHODS = ('dummy_method')

def simulate(run_name : str, global_config : GlobalConfig, body_config : SystemState) -> None:
    assert global_config.method in METHODS

    bar = common.get_my_progressbar(global_config.iter_num)
    for do_write, iter_idx in common.get_iter_indices(global_config.iter_num, global_config.output_point_num):
        if do_write:
            common.write_body_config(run_name, body_config, iter_idx)
            bar.update(iter_idx)
