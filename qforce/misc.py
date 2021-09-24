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


def check_if_file_exists(file):
    if not os.path.exists(file) and not os.path.exists(f'{file}_qforce'):
        sys.exit(f'ERROR: "{file}" does not exist.\n')
    return file

def check_continue(config):
    if config.general.debug_mode:
        x = input('\nDo you want to continue y/n? ')
        if x not in ['yes', 'y', '']:
            print()
            sys.exit(0)
