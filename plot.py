import argparse
import logging
from pathlib import Path

from nbody_sim import common, plot


def setup_logging(is_debug : bool) -> None:
    loglevel = logging.DEBUG if is_debug else logging.INFO
    logging.basicConfig(level=loglevel)


def create_subparser(subparsers, name):
    parser = subparsers.add_parser(name)
    parser.add_argument('run_name', type=str)
    parser.set_defaults(action_name=name)
    return parser


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save', type=Path, help='save the image to this file')
    subparsers = parser.add_subparsers()

    parser_r = create_subparser(subparsers, 'pos')
    parser_r.add_argument('--bodies', type=str, nargs='*', help='names of bodies to plot')

    parser_energy = create_subparser(subparsers, 'energy')
    parser_energy.add_argument('--bodies', type=str, nargs='*', help='names of bodies to plot separately')
    parser_energy.add_argument('--cumulative', action='store_true', help='do cumulative plot')

    parser_momentum = create_subparser(subparsers, 'momentum')
    parser_momentum.add_argument('--bodies', type=str, nargs='*', help='names of bodies to plot separately, cumulative if not specified')
    parser_momentum.add_argument('--cumulative', action='store_true', help='do cumulative plot')

    args = parser.parse_args()

    setup_logging(args.debug)
    logging.debug(str(args))

    if args.action_name == 'pos':
        plot.plot_positions(args.run_name, args.save, args.bodies)
    elif args.action_name == 'energy':
        plot.plot_energy(args.run_name, args.save, args.bodies, args.cumulative)
    elif args.action_name == 'momentum':
        plot.plot_momentum(args)
    else:
        raise RuntimeError(f'unexpected plotting action: {args.action_name}')


if __name__ == '__main__':
    main()
