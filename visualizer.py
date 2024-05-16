import sys
from utils.vis import Visualizer
import argparse

def main():
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-g', '--groundtruth', type=str, default=None, help="filename")
    arg_parser.add_argument('-p', '--prediction', type=str, default=None, help="filename")
    arg_parser.add_argument('-s', '--start', type=float, default=0, help="subsample the datapoints starting from x percent time")
    args = arg_parser.parse_args()

    
    if not (args.groundtruth or args.prediction):
        arg_parser.usage = arg_parser.format_help()
        arg_parser.print_usage()
    else:
        visualizer = Visualizer(args.groundtruth, args.prediction)
        visualizer.run(startTimePercent = args.start)

if __name__ == "__main__":
    main()
