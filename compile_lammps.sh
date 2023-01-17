cd pair_allegro && bash patch_lammps.sh ../lammps/
cd ..
cd lammps && rm -rf build && mkdir build  && cd build && cmake ../cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=../../libtorch -DMKL_INCLUDE_DIR=`python -c "import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()[\"include\"]).parent)"` && make