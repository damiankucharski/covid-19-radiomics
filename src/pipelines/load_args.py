def load_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-f', '--features')
    parser.add_argument('-m', '--metadata')
    parser.add_argument('-s', '--suffix')
    return parser.parse_args()