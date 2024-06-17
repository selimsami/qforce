import subprocess

def compute_gdma(job, fchk_file):

    write_gdma(job, fchk_file)
    run_gdma(job)
    charges, dipoles, quadrupoles = read_gdma(job)
    return charges, dipoles, quadrupoles


def write_gdma(job, fchk_file):

        input_file = f'''
Title "{job.name} - GDMA input"
File {fchk_file}

Angstrom
Multipoles
  switch 4
  Limit 2
  Limit 2 H 
  Radius H 0.325
  Punch {job.name}.punch
Start

Finish'''

        with open(f'{job.dir}/gdma_input', 'w') as file:
            file.write(input_file)


def run_gdma(job):

    with open(f'{job.dir}/gdma_input', 'r') as inp:
        with open(f'{job.dir}/gdma_result', 'w') as out:
            pop = subprocess.Popen(['/Users/ssami/gdma/bin/gdma'],
                                   stdin=inp, stderr=out, stdout=out)
    pop.wait()


def read_gdma(job):
    charges, dipoles, quadrupoles = [], [], []

    with open(f'{job.dir}/gdma_result', 'r') as file:
        for line in file:
            if ' Maximum rank =' in line:
                charges.append(0.)
                dipoles.append([0.]*3)
                quadrupoles.append([0.]*5)

            if 'Q00  =' in line:
                charges[-1] = float(line.split()[2])
                line = file.readline()

                if '|Q1| =' in line:
                    split = line.split()
                    if 'Q10' in line:
                        i = split.index('Q10')
                        dipoles[-1][2] = float(split[i+2])
                    if 'Q11c' in line:
                        i = split.index('Q11c')
                        dipoles[-1][0] = float(split[i+2])
                    if 'Q11s' in line:
                        i = split.index('Q11s')
                        dipoles[-1][1] = float(split[i+2])

                    line = file.readline()

                if '|Q2| =' in line:
                    split = line.split()
                    if 'Q20' in line:
                        i = split.index('Q20')
                        quadrupoles[-1][0] = float(split[i+2])
                    if 'Q21c' in line:
                        i = split.index('Q21c')
                        quadrupoles[-1][1] = float(split[i+2])
                    if 'Q21s' in line:
                        i = split.index('Q21s')
                        quadrupoles[-1][2] = float(split[i+2])
                    if 'Q22c' in line:
                        i = split.index('Q22c')
                        quadrupoles[-1][3] = float(split[i+2])
                    if 'Q22s' in line:
                        i = split.index('Q22s')
                        quadrupoles[-1][4] = float(split[i+2])

                quadrupoles[-1] = convert_quads_to_cartesian(quadrupoles[-1])

            if 'Total multipoles ' in line:
                break

    return charges, dipoles, quadrupoles


def convert_quads_to_cartesian(quads):
    q_11 = -0.5 * quads[0] + 3**0.5 / 2 * quads[3]
    q_22 = -0.5 * quads[0] - 3**0.5 / 2 * quads[3]
    q_33 = quads[0]
    q_12 = 3**0.5 / 2 * quads[4]
    q_13 = 3**0.5 / 2 * quads[1]
    q_23 = 3**0.5 / 2 * quads[2]
    return [q_11, q_22, q_33, q_12, q_13, q_23]
