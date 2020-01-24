import logging
from . import common

module_logger = logging.getLogger(__name__)

def parse_and_run(run_name : str) -> None:
    """The entry point."""
    module_logger.debug(f'Running simulation "{run_name}"')

    global_config = common.read_global_config(run_name)
    body_config = common.read_body_config(run_name)
    module_logger.debug(f'Global config: {global_config}')
    module_logger.debug(f'Body config: {body_config}')
