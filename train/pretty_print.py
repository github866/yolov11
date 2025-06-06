import os
import sys

# ANSI color codes for pretty output
RESET = '\033[0m'
BOLD = '\033[1m'
BLUE = '\033[94m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
MAGENTA = '\033[95m'


def pretty_print_dir(path: str = '.', prefix: str = ''):
    """
    Recursively prints the directory structure in a pretty tree format.
    """
    entries = sorted(os.listdir(path))
    entries_count = len(entries)
    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        connector = '└── ' if i == entries_count - 1 else '├── '
        if os.path.isdir(full_path):
            color = BLUE + BOLD
            print(f"{prefix}{connector}{color}{entry}/{RESET}")
            extension = '    ' if i == entries_count - 1 else '│   '
            pretty_print_dir(full_path, prefix + extension)

            color = GREEN if entry.endswith('.py') else CYAN if entry.endswith('.sh') else YELLOW if entry.endswith('.md') else MAGENTA if entry.endswith('.yaml') else RESET        else:
            print(f"{prefix}{connector}{color}{entry}{RESET}")

if __name__ == '__main__':
    print(f"{BOLD}{os.path.basename(os.path.abspath('.'))}/{RESET}")
    pretty_print_dir('.')
