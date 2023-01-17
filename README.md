
![image](logo.png)
This repository implements an extension of the [Allegro](https://github.com/mir-group/allegro), **Allegro-Legato** (meaning fast and smooth).
Allegro-Legato provide a neural-network molecular dynamics (MD) simulator with enhanced robustness, i.e., the MD simulation runs for a long time without failure.

This legato extension is developed by [Hikaru Ibayashi](https://viterbi-web.usc.edu/~ibayashi/).
If you have questions about this repository, feel free to reach out to me (ibayashi[at]usc.edu).


## Installation
Note: This implementation assumes HPC environment.
Our implementation is based on `allegro` package.
Please refer [Allegro installation instructions](https://github.com/mir-group/allegro#installation) to see the detailed dependencies.


Required environment (with versions confirmed to work.)
- gcc: 8.3.0
- git: 2.25.0
- cmake: 3.16.2
- cuda: 11.3.0
- Python: 3.10.8
- Torch: 1.11.0
- wandb: 0.13.5

With those external modules and libraries installed, run the following command to install the nequip library and allegro library with legato extension.
```bash
cd nequip && pip install .
```
```bash
cd allegro &&  pip install .
```

Then, you run the following to compile LAMMPS environment.
```bash
bash compile_lammps.sh
```
## Execution
We provide three features to build and try Allegro-Legato:
- Training: `python train.py`
- Measuring sharpness: `python measure_sharpness.py`
- Measuring $t_\text{failure}$: `python measure_t_failure.py`

Note: $t_\text{failure}$ mean the number of steps until the simulation breaks. Allegro-Legato has larger $t_\text{failure}$ than the original Allegro, i.e., more robust.
### Start Minimally
To check if those three features work on your environment, first run the following command as a sanity-check. 

```bash
python train.py --sanity-check;
python measure_sharpness.py --sanity-check;
python measure_t_failure.py --sanity-check;
```

