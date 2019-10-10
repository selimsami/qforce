import os, subprocess, shutil

def make_dihedral_scan_input(inp, frag_id, id_no):
    out_dir = f'{inp.job_name}_qforce/fragments'
    out_file = f'{out_dir}/{frag_id}~{id_no}.com'
    xyz_file = f'{inp.frag_lib}/{frag_id}/coords_{id_no}.xyz'

    os.makedirs(out_dir, exist_ok=True)
    run_obabel(xyz_file, f'{out_dir}/{out_file}', "gzmat")

    
    








def make_hessian_input(inp, inp_file = None, out_file = None):
    """
    Scope:
    ------
    
    Output:
    -------
    
    
    """
#    inp, out, out_path = get_names(inp)
#    if not os.path.exists(out[0]):
#        os.makedirs(out[0])
    # for each different input type, create the necessary formatting

    out_dir = f'{inp.job_name}_qforce'
    os.makedirs(out_dir, exist_ok=True)
#    
#    if inp.job_type == "input_hessian":
#        out_file = f'{inp.job_name}_hessian'
#        run_obabel(inp.coord_file, f'{out_dir}/{out_file}', "com")
#        change_run_settings(inp, out, out_path)
#    if inp.job_type == "fragment":
#        out_dir =  f'{out_dir}/fragments'
#        os.makedirs(out_dir, exist_ok=True)
#        run_obabel(inp_file, f'{out_dir}/{out_file}', "gzmat")
#        
        
#        add_dihedrals(inp, out, out_path)

    
def get_names(inp):
    inp.job = inp.job_type.split("_")[1]
    inp.base = os.path.splitext(inp.coord_file)[0]
    inp.key = {"dihedral": "opt=modredundant ", "hessian": "freq opt ",
               "traj": ""} 
    if inp.job == "traj":
        out = [f"{inp.base}_traj/"]
    else:
        out = [f"{inp.base}_qm_inputs/"]
    out.append(f"{inp.base}_{inp.job}")
    if  (len(inp.pre_input_commands) or len(inp.post_input_commands)) != 0:
        out.append(".inp")
    else:
        out.append(".com")
    out_path = "{}{}{}".format(*out)
    return inp, out, out_path
       
def run_obabel(inp_file, out_file, out_type):
    obabel = subprocess.Popen(['obabel', inp_file, "-o", out_type, "-O",
                               out_file, "-unique"], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    obabel.wait()

def add_dihedrals(inp, out, out_path):
    out_orig, orig_path = out[1], out_path
    for dihed in inp.dihedrals:
        out[1] = "{}_{}_{}_{}_{}".format(out_orig, *dihed)
        out_path = "{}{}{}".format(*out)
        shutil.copy2(orig_path, out_path)
        with open(out_path, "a") as scan:
            scan.write("D {} {} {} {} S {} {}\n\n".format(*dihed, inp.scan_no, 
                                                          inp.scan_step))
        change_run_settings(inp, out, out_path)
    os.remove(orig_path)
    
def change_run_settings(inp, out, out_path):
    with open(out_path, "r") as file:
        coord =  file.readlines()
    title = (f"{out[1]}\n")
    if inp.disp != "":
        disp = f'EmpiricalDispersion={inp.disp}'   
    
    coord = coord[5:]
    with open(out_path, "w") as file:
        for line in inp.pre_input_commands:
            if "<outfile>" in line:
                on, off = line.index("<"), line.index(">") +1
                file.write(f"{line[:on]}{out[1]}{line[off:]}\n")
            else:
                file.write(f"{line}\n")
        file.write(f"{inp.nproc}{inp.mem}")
        file.write(f"%chk={out[1]}.chk\n")
        file.write(f"#{inp.key[inp.job]} {inp.method} {inp.basis} {disp}"
                   f" pop={inp.charge_method} \n\n")
        file.write(f"{title}\n")
        file.write(f"{inp.charge} {inp.multi}\n")
        for line in coord:
            file.write(line)
        for line in inp.post_input_commands:
            file.write(line)