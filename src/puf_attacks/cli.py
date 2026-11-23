import argparse
from .experiments import run
from .analysis import summarize


def main():
    parser = argparse.ArgumentParser(description='PUF modeling attacks')
    commands = parser.add_subparsers(dest='command',required=True)
    experiment = commands.add_parser('run')
    experiment.add_argument('--config',required=True)
    experiment.add_argument('--output',required=True)
    experiment.add_argument('--limit',type=int,help='run at most this many new jobs; resume with same command')
    analysis = commands.add_parser('summarize')
    analysis.add_argument('directory')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.config,args.output,args.limit)
    else:
        summarize(args.directory)


if __name__ == '__main__':
    main()
