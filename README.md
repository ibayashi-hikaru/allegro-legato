![image](logo.png)

<br>

https://user-images.githubusercontent.com/15017849/224492839-5fc1c686-60ab-424f-81dd-623520e9afad.mp4


![aaa](demo.gif)

<br><br>
This repository implements an extension of the [Allegro](https://github.com/mir-group/allegro), Allegro-Legato, which provides a neural-network molecular dynamics (MD) simulator with enhanced robustness.
Allegro-Legato realizes the MD simulation for a long time without failure.
This legato extension is developed by [Hikaru Ibayashi](http://hikaru-ibayashi.com/).
If you have questions about this repository, feel free to reach out to me (ibayashi[at]usc.edu).


## Installation
Note: This implementation assumes an HPC environment.
Required environment (with versions confirmed to work.)
- gcc: 8.3.0
- git: 2.25.0
- cmake: 3.16.2
- cuda: 11.3.0
- Python: 3.10.8
- wandb: 0.13.5

With those external modules and libraries installed, run the following commands to install the nequip library and allegro library with the legato extension.
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
```bash
cd nequip && pip install .
```
```bash
cd allegro &&  pip install .
```
Then, you run the following to compile the LAMMPS environment.
```bash
bash compile_lammps.sh
```
Please refer [Allegro installation instructions](https://github.com/mir-group/allegro#installation) to see the detailed dependencies.
## Usage
We provide the following three features. Simply execute the python scripts and you can enjoy each feature parallelized on HPC.  
- Training: `python train.py`
- Measuring sharpness: `python measure_sharpness.py`
- Measuring $t_\text{failure}$: `python measure_t_failure.py`

Note: $t_\text{failure}$ means the number of steps until the simulation breaks. Allegro-Legato has larger $t_\text{failure}$ than the original Allegro, i.e., more robust.
### Start Minimally
To check if those three features work on your environment, first run the following command as a sanity check. 

```bash
python train.py --sanity-check;
python measure_sharpness.py --sanity-check;
python measure_t_failure.py --sanity-check;
```
