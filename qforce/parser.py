import argparse
import os
import sys


def check_if_file_exists(file):
    if not os.path.exists(file) and not os.path.exists(f'{file}_qforce'):
        sys.exit(f'ERROR: "{file}" does not exist.\n')
    return file


def parse():
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument('f', type=check_if_file_exists, metavar='file',
                        help=('Input coordinate file mol.ext (ext: pdb, xyz, gro, ...)\n'
                              'or directory (mol or mol_qforce) name.'))
    parser.add_argument('-o', type=check_if_file_exists, metavar='options',
                        help='File name for the optional options.')
    args = parser.parse_args()

    return args
