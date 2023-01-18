from hpc_utilities import Sbatch, JobPool
import os
import time
import secrets
import glob
import numpy as np
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
def lammps_script(pot, ntasks, temp, timestep):
    script = []
    id=f"{secrets.token_hex(16)}"
    script.append(f"mpirun -np {ntasks}")
    script.append(f"{root_dir}/lammps/build/lmp")
    script.append(f"-in {root_dir}/lammps_config/nh3/lmp.in")
    script.append(f"-var temp {temp}")
    script.append(f"-var dat {root_dir}/lammps_config/nh3/nh3.data")
    script.append(f"-var elem \"H N\"")
    script.append(f"-var pot {root_dir}/results/{pot}/deployed.pth")
    script.append(f"-var suffix {id}")
    script.append(f"-var timestep {timestep}")
    script.append(f"-log {root_dir}/results/{pot}/partial_t_failure_{id}.dat")
    return " ".join(script)

def submit_lammps_job(pot, ntasks, temp, timestep):
    sbatch = Sbatch()
    sbatch.header(ntasks=ntasks, gpu_type="a100", account="your_account")
    sbatch.add_command(f"cd {root_dir}/results/{pot};")
    sbatch.add_command(lammps_script(pot, ntasks, temp, timestep))
    return sbatch.submit_job()

def dump_t_failure(pot):
    t_falure_list = []
    for file_name in glob.glob(f"results/{pot}/partial_t_failure*"):
        with open(file_name, 'r') as f:
            last_line = f.readlines()[-1]
            t_falure = int(last_line.split()[0])
            t_falure_list.append(t_falure)
    for file_name in glob.glob(f"results/{pot}/partial_t_failure*"):
        os.system(f"rm {file_name}")
    with open(f"results/{pot}/t_failure.dat", 'w') as f:
        f.write(f"mean: {np.mean(t_falure_list)}\n")
        f.write(f"std: {np.std(t_falure_list)}\n")

if __name__ == "__main__":
    pots = ["nh3/L0", "nh3/L1", "nh3/L2",
            "nh3/legato_L1_rho_0_001",
            "nh3/legato_L1_rho_0_0025",
            "nh3/legato_L1_rho_0_005",
            "nh3/legato_L1_rho_0_01",
            "nh3/legato_L1_rho_0_025",
            "nh3/legato_L1_rho_0_05"]

    if sys.argv[1] == "--sanity-check":
        pots = ["nh3/sanity_checker", "nh3/sanity_checker_legato"]
    job_pools = [JobPool(pot) for pot in pots]
    sample_num = 10
    # Submit jobs
    for job_pool in job_pools:
        for file_name in glob.glob(f"results/{job_pool.pot}/partial_t_failure*"):
            os.system(f"rm {file_name}")
        os.system(f"rm results/{job_pool.pot}/t_failuire.dat")
        train_dir = f"results/{job_pool.pot}"
        deploy_dir = f"results/{job_pool.pot}"
        os.system(f"nequip-deploy build --train-dir {train_dir} {deploy_dir}/deployed.pth;")
        for _ in range(sample_num):
            job_id = submit_lammps_job(pot=job_pool.pot,
                                       ntasks=32,
                                       temp=200,
                                       timestep=0.0025)
            job_pool.add_job(job_id)
        job_pool.submitted = True
    # Wait until all jobs are done and dump results
    while True:
        os.system(f"clear")
        print(f"Mearuring t_failure")
        print("=====================================")
        for job_pool in job_pools:
            job_pool.show_job_status()
            if job_pool.done:
                dump_t_failure(job_pool.pot)
                job_pools.remove(job_pool)
        time.sleep(5)
        if len(job_pools) == 0:
            break

    os.system(f"clear")
    for pot in pots:
        print(f"Pot: {pot}")
        os.system(f"cat results/{pot}/t_failure.dat")
        print()
