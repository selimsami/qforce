import argparse, os, sys

def check_input_file(file):
    if not os.path.isfile(file):
        sys.exit(f'ERROR: Input file with the name "{file}" does not exist.\n')
    return file

def parse():
    options = ["dihedralfitting", "fragment", "input_hessian", "input_traj",
               "input_dihedral", "bondangle", "dipolefitting","polarize",
               "hessianfitting"]
    opt_help = 'Possible job types for Q-Force:\n - ' + '\n - '.join(options)

    parser = argparse.ArgumentParser(formatter_class=
                                     argparse.RawTextHelpFormatter)
    parser.add_argument('job_type', metavar='job_type', choices=options,
                        help=opt_help)
    parser.add_argument('input_file', type=check_input_file, 
                        help='Input file name. (PBB/GRO)')
    args = parser.parse_args()
    return args.job_type, args.input_file