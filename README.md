![image](logo.png)

<br>

<p align="center">
  <img src="https://user-images.githubusercontent.com/15017849/224515542-7f902174-6040-4c3f-8ca9-7a8df52253e5.gif" alt="animated" />
</p>

<br>

This repository implements an extension of the [Allegro](https://github.com/mir-group/allegro), Allegro-Legato, which provides a neural-network molecular dynamics (MD) simulator with enhanced robustness.
Allegro-Legato realizes the MD simulation for a long time without failure.
This legato extension is developed by [Hikaru Ibayashi](https://hikaru-ibayashi.com/).

When you use Allegro-Legato in your paper, please use the following BibTeX to cite.
```
@inproceedings{ibayashi2023allegro,
  title={Allegro-Legato: Scalable, Fast, and Robust Neural-Network Quantum Molecular Dynamics via Sharpness-Aware Minimization},
  author={Ibayashi, Hikaru and Razakh, Taufeq Mohammed and Yang, Liqiu and Linker, Thomas and Olguin, Marco and Hattori, Shinnosuke and Luo, Ye and Kalia, Rajiv K and Nakano, Aiichiro and Nomura, Ken-ichi and others},
  booktitle={International Conference on High Performance Computing},
  pages={223--239},
  year={2023},
  organization={Springer}
}
```
- [https://arxiv.org/abs/2303.08169](https://arxiv.org/abs/2303.08169)
- [https://link.springer.com/chapter/10.1007/978-3-031-32041-5_12](https://link.springer.com/chapter/10.1007/978-3-031-32041-5_12)



If you have questions about this repository, feel free to contact me (ibayashi[at]alumni.usc.edu).

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
We provide the following three features. Simply execute the Python scripts and you can enjoy each feature parallelized on HPC.  
- Training: `python train.py`
- Measuring sharpness: `python measure_sharpness.py`
- Measuring $t_\text{failure}$: `python measure_t_failure.py`

Note: $t_\text{failure}$ means the number of steps until the simulation breaks. Allegro-Legato has larger $t_\text{failure}$ than the original Allegro, i.e., more robust.
### Start Minimally
To check if those three features work on your environment, first, run the following command as a sanity check. 

```bash
python train.py --sanity-check;
python measure_sharpness.py --sanity-check;
python measure_t_failure.py --sanity-check;
```


## Long term simulation
Since this code is intended for the short-term simulation to tune the hyper-parameter, $\rho$, (Sec. 2.3), simulations terminate after the system temperature deviates by more than 100 degree.

However, if you want to continue running your simulation as long as possible, please comment out [this line](https://github.com/ibayashi-hikaru/allegro-legato/blob/main/lammps/src/utils.cpp#L154C29-L154C29) and recompile LAMMPS.





