import sys
import os
Path = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(Path)


def get_path(ph):
    return os.path.join(Path, ph)
