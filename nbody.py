import argparse
import logging
from nbody_sim import parse_and_run

def setup_logging(is_debug : bool) -> None:
    loglevel = logging.DEBUG if is_debug else logging.INFO
    logging.basicConfig(level=loglevel)

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('simulation_name', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    setup_logging(args.debug)
    parse_and_run(args.simulation_name)

if __name__ == '__main__':
    main()
