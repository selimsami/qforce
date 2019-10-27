from ase.io import read
import networkx as nx
from .elements import elements


def make_qm_input(inp, G, out_file):
    if inp.job_type == 'fragment':
        out_dir = inp.frag_dir
    else:
        out_dir = inp.job_dir

    out_path = change_run_settings(inp, out_dir, out_file, G)
    return out_path


def make_hessian_input(inp):
    G = nx.Graph()
    molecule = read(inp.xyz_file)
    coords = molecule.get_positions()
    atomic_no = molecule.get_atomic_numbers()

    for i, (c, a) in enumerate(zip(coords, atomic_no)):
        G.add_node(i, elem=a, coords=c)

    out_path = make_qm_input(inp, G, f'{inp.job_name}_hessian')

    print("Creating the input file for the hessian calculation in:")
    print(f"{out_path}\n")
    print("Please run the calculation and put the output (.fchk and out/log) "
          "files in the same directory.")


def change_run_settings(inp, out_dir, out_file, G):
    e = elements()
    key = {"fragment": "opt=modredundant ", "init": "freq opt"}
    out_path = f'{out_dir}/{out_file}'

    title = out_file

    if len(inp.pre_input_commands) + len(inp.post_input_commands) > 0:
        out_path = f'{out_path}.inp'
    else:
        out_path = f'{out_path}.com'
    if inp.disp != "":
        inp.disp = f' EmpiricalDispersion={inp.disp}'

    if inp.job_type == 'init':
        inp.charge_method = f'({inp.charge_method}, NBOREAD)'

    with open(out_path, "w") as file:
        for line in inp.pre_input_commands:
            if "<outfile>" in line:
                on, off = line.index("<"), line.index(">") + 1
                file.write(f"{line[:on]}{title}{line[off:]}\n")
            else:
                file.write(f"{line}\n")
        file.write(f"{inp.nproc}{inp.mem}")
        file.write(f"%chk={title}.chk\n")
        file.write(f"#{key[inp.job_type]} {inp.method} {inp.basis}{inp.disp}"
                   f" pop={inp.charge_method} \n\n")
        file.write(f"{title}\n\n")
        file.write(f"{inp.charge} {inp.multi}\n")

        for data in sorted(G.nodes.data()):
            atom, [c1, c2, c3] = e.sym[data[1]['elem']], data[1]['coords']
            file.write(f'{atom:>3s} {c1:>12.6f} {c2:>12.6f} {c3:>12.6f}\n')

        if inp.job_type == "fragment":
            file.write("\nD {} {} {} {} S {} {}\n\n".format(*G.graph['scan'],
                                                            inp.scan_no,
                                                            inp.scan_step))
        elif out_path.endswith('.inp'):
            file.write('\n\$nbo BNDIDX \$end\n\n')
        else:
            file.write('\n$nbo BNDIDX $end\n\n')
        for line in inp.post_input_commands:
            file.write(line)
    return out_path
