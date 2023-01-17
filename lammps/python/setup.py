# this only installs the LAMMPS python package
# it assumes the LAMMPS shared library is already installed
from distutils.core import setup
from sys import version_info
import os,time

LAMMPS_PYTHON_DIR = os.path.dirname(os.path.realpath(__file__))
LAMMPS_DIR = os.path.dirname(LAMMPS_PYTHON_DIR)
LAMMPS_SOURCE_DIR = os.path.join(LAMMPS_DIR, 'src')

if not os.path.exists(LAMMPS_SOURCE_DIR):
    # allows installing and building wheel from current directory
    LAMMPS_DIR = os.path.realpath(os.path.join(os.environ['PWD'], '..'))
    LAMMPS_SOURCE_DIR = os.path.join(LAMMPS_DIR, 'src')

def get_lammps_version():
    version_h_file = os.path.join(LAMMPS_SOURCE_DIR, 'version.h')
    with open(version_h_file, 'r') as f:
        line = f.readline()
        start_pos = line.find('"')+1
        end_pos = line.find('"', start_pos)
        t = time.strptime("".join(line[start_pos:end_pos].split()), "%d%b%Y")
        return "{}.{}.{}".format(t.tm_year,t.tm_mon,t.tm_mday)

if version_info.major >= 3:
    pkgs = ['lammps', 'lammps.mliap']
else:
    pkgs = ['lammps']

setup(
    name = "lammps",
    version = get_lammps_version(),
    author = "Steve Plimpton",
    author_email = "sjplimp@sandia.gov",
    url = "https://www.lammps.org",
    description = "LAMMPS Molecular Dynamics Python package",
    license = "GPL",
    packages=pkgs,
)
