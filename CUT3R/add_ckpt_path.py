import sys
import os
import os.path as path


def add_path_to_dust3r(ckpt):
    HERE_PATH = os.path.dirname(os.path.abspath(ckpt))
    PROJECT_ROOT = os.path.dirname(HERE_PATH)
    for p in (PROJECT_ROOT, HERE_PATH):
        if p not in sys.path:
            sys.path.insert(0, p)
