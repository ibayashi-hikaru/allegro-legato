from hpc_utilities import Sbatch, JobPool
import numpy as np
import os
import time
import glob
import sys


def submit_measure_sharpness_jobs(train_dir, epsilon, sample_num, parallel_num):
    for file_name in glob.glob(f"{train_dir}/partial_sharpness*.dat"):
        os.system(f"rm {file_name}")
    os.system(f"rm results/{job_pool.pot}/sharpness.dat")
    sbatch = Sbatch()
    sbatch.header(ntasks=4, gpu_type="a100")
    sbatch.add_command(
        f"python nequip/nequip/scripts/measure_sharpness.py --train-dir {train_dir} --epsilon {epsilon} --sample-num {int(sample_num / parallel_num)}")
    job_ids = []
    for _ in range(parallel_num):
        job_id = sbatch.submit_job()
        job_ids.append(int(job_id))
    return job_ids

def dump_sharpness(train_dir):
    # Concatenate all the files in the directory into a single string
    concatenated_string = ""
    for file_name in glob.glob(f"{train_dir}/partial_sharpness*.dat"):
        with open(f"{file_name}", "r") as f:
            concatenated_string += f.read()
    for file_name in glob.glob(f"{train_dir}/partial_sharpness*.dat"):
        os.system(f"rm {file_name}")
    data = np.fromstring(concatenated_string,
                         dtype=float, sep=" ")
    # Write the concatenated string to a new file
    with open(f"{train_dir}/sharpness.dat", "w") as f:
        f.write(f"{data.mean()}")

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
    epsilon = 0.05
    sample_num = 100
    parallel_num = 10 

    job_pools = [JobPool(pot) for pot in pots]
    # Submit jobs
    for job_pool in job_pools:
        job_ids = submit_measure_sharpness_jobs(epsilon=epsilon,
                                                sample_num=sample_num,
                                                parallel_num=parallel_num,
                                                train_dir=f"results/{job_pool.pot}")
        job_pool.add_jobs(job_ids)
        job_pool.submitted = True
    # Wait until all jobs are done and dump results
    while True:
        os.system(f"clear")
        print(f"Mearuring sharpness")
        print("=====================================")
        for job_pool in job_pools:
            job_pool.show_job_status()
            if job_pool.done:
                dump_sharpness(f"results/{job_pool.pot}")
                job_pools.remove(job_pool)
        time.sleep(5)
        if len(job_pools) == 0:
            break

    os.system(f"clear")
    for pot in pots:
        print(f"Pot: {pot}")
        os.system(f"cat results/{pot}/sharpness.dat")
        print()