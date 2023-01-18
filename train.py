from hpc_utilities import Sbatch, JobPool
import sys

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

    job_pool = JobPool(f"Training")
    for pot in pots:
        sbatch = Sbatch()
        # The very first run need many nodes and large memory to preprocess data
        sbatch.header(ntasks=32, gpu_type="a100", memory=128, account="your_account")
        sbatch.add_command(f"rm -rf results/{pot};")
        sbatch.add_command(f"nequip-train allegro/configs/{pot}.yaml;")
        id = sbatch.submit_job()
        job_pool.add_job(id)
    job_pool.submitted = True
    job_pool.wait_until_all_done()

    # Some of the training jobs may exceed 24h, so we need to resubmit them
    job_pool = JobPool(f"Training")
    for pot in pots:
        sbatch = Sbatch()
        sbatch.header(ntasks=4, gpu_type="a100", memory=32, account="your_account")
        sbatch.add_command(f"nequip-train allegro/configs/{pot}.yaml;")
        id = sbatch.submit_job()
        job_pool.add_job(id)
    job_pool.submitted = True
    job_pool.wait_until_all_done()