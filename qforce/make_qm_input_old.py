import os, subprocess, shutil

def make_qm_input(inp):
    """
    Scope:
    ------
    Make QM (so far, only Gaussian 09/16) input for various different
    calculations.
    
    Output:
    -------
    
    
    """
    inp, out, out_path = get_names(inp)
    if not os.path.exists(out[0]):
        os.makedirs(out[0])
    # for each different input type, create the necessary formatting
    if inp.job == "traj":
        run_obabel(inp, out_path, "com", "-m")
        for file in os.listdir(out[0]):
            name, ext = os.path.splitext(file)
            if ext == ".com" or ext == ".inp":
                out[1] = name
                out_path = "{}{}{}".format(*out)
                change_run_settings(inp, out, out_path)
                inp.key["traj"] = ""
    elif inp.job == "dihedral":
        run_obabel(inp, out_path, "gzmat", "-unique")
        add_dihedrals(inp, out, out_path)
    
    else:
        run_obabel(inp, out_path, "com", "-unique")
        change_run_settings(inp, out, out_path)
    print(f"Input files are in the directory:\n{out[0]}")
    
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
       
def run_obabel(inp, out_path, out_type, arg):
    obabel = subprocess.Popen(['obabel', inp.coord_file, "-o", out_type, "-O",
                               out_path, arg], stdout=subprocess.PIPE,
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
    if inp.job == "traj":
        title = coord[2]
    else:
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
