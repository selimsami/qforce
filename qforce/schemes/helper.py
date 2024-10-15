from ase.io import read


def coords_ids_iter(filename):
    if filename is None:
        return
    molecules = read(filename, index=':')
    for molecule in molecules:
        yield molecule.get_positions(), molecule.get_atomic_numbers()


def write_xyz(filename, ids, coords, comment=None):
    with open(filename, 'w') as fh:
        if comment is None:
            comment = ''
        comment = ' -- '.join(comment.splitlines())
        fh.write(f'{len(ids)}\n{comment}\n')
        for id, (x, y, z) in zip(ids, coords):
            fh.write(f"{id}  {x:12.8f} {y:12.8f} {z:12.8f}\n")
