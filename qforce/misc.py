import os
import sys

LOGO = """
          ____         ______
         / __ \       |  ____|
        | |  | |______| |__ ___  _ __ ___ ___
        | |  | |______|  __/ _ \| '__/ __/ _ \\
        | |__| |      | | | (_) | | | (_|  __/
         \___\_\      |_|  \___/|_|  \___\___|

                     Selim Sami
            University of Groningen - 2020
            ==============================
"""

LOGO_SEMICOL = """
;          ____         ______
;         / __ \       |  ____|
;        | |  | |______| |__ ___  _ __ ___ ___
;        | |  | |______|  __/ _ \| '__/ __/ _ \\
;        | |__| |      | | | (_) | | | (_|  __/
;         \___\_\      |_|  \___/|_|  \___\___|
;
;                     Selim Sami
;            University of Groningen - 2020
;            ==============================
"""


def check_if_file_exists(filename):
    if not os.path.exists(filename) and not os.path.exists(f'{filename}_qforce'):
        raise ValueError(f'"{filename}" does not exist.\n')
    return filename
