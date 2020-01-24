import logging

module_logger = logging.getLogger(__name__)

def parse_and_run(run_name : str) -> None:
    module_logger.debug(f'Running simulation "{run_name}"')
