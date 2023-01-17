import subprocess
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import glob
import secrets


class JobPool:
    def __init__(self, pot) -> None:
        self.job_states = {}
        self.pot = pot
        self.process = None
        self.submitted = False
        self.done = False

    class State(Enum):
        PENDING = 1
        RUNNING = 2
        DONE = 3

    def add_job(self, job):
        if not self.submitted:
            self.job_states[job] = self.State.PENDING
        else:
            raise Exception("Jobs are already submitted")

    def add_jobs(self, job_list):
        if not self.submitted:
            for job in job_list:
                self.job_states[job] = self.State.PENDING
        else:
            raise Exception("Jobs are already submitted")

    def refresh_status(self):
        for job in self.job_states:
            # Check the exit code
            if self.job_states[job] == self.State.DONE:
                continue
            try:
                stdout = subprocess.check_output(["squeue", "-j", str(job)])
                stdout_list = stdout.decode("utf-8").split("\n")
                job_status = stdout_list[1].split()[4]
                # The command was successful
                if job_status == "PD":
                    self.job_states[job] = self.State.PENDING
                elif job_status == "R":
                    self.job_states[job] = self.State.RUNNING
            except:
                self.job_states[job] = self.State.DONE

    def get_job_status(self, job):
        self.refresh_status()
        return self.job_states[job]

    def show_job_status(self):
        if not self.submitted:
            print("Jobs are not submitted yet")
        else:
            self.refresh_status()
            done_jobs = len(
                [job for job in self.job_states if self.job_states[job] == self.State.DONE])
            pending_jobs = len(
                [job for job in self.job_states if self.job_states[job] == self.State.PENDING])
            running_jobs = len(
                [job for job in self.job_states if self.job_states[job] == self.State.RUNNING])
            print(f"{self.pot}")
            print(
                f"\t Done: {done_jobs}, Running: {running_jobs}, Pending: {pending_jobs}")
            if all([job_status == self.State.DONE for job_status in self.job_states.values()]):
                self.done = True

    def wait_until_all_done(self):
        while True:
            self.refresh_status()
            os.system("clear")
            self.show_job_status()
            if all([job_status == self.State.DONE for job_status in self.job_states.values()]):
                break
            time.sleep(5)

    def __del__(self):
        for job in self.job_states:
            os.system(f"scancel {job}")
            os.system(f"rm slurm-{job}.out")

class Sbatch:
    def __init__(self) -> None:
        self.sbatch_name = ""
        self.contents_list = []

    def header(self, gpu_type="a100", memory=64, ntasks=64,
               account="anakano_81", gpu_num=1, partition="gpu", time="23:59:59"):
        header_contents = []
        header_contents.append(f"#!/bin/bash")
        header_contents.append(f"#SBATCH --account={account}")
        header_contents.append(f"#SBATCH --gres=gpu:{gpu_type}:1")
        header_contents.append(f"#SBATCH --partition={partition}")
        header_contents.append(f"#SBATCH --time={time}")
        header_contents.append(f"#SBATCH --mem={memory}GB")
        header_contents.append(f"#SBATCH --ntasks={ntasks}")
        header_contents.append(f"export OMP_NUM_THREADS={ntasks}")
        header = "\n".join(header_contents)
        self.contents_list.append(header)

    def add_command(self, command):
        self.contents_list.append(command)

    def submit_job(self):
        if self.sbatch_name == "":
            file_name = f"sbatch_{secrets.token_hex(16)}.sh"
        else:
            file_name = f"{self.sbatch_name}_{secrets.token_hex(16)}.sh"
        with open(file_name, "w") as f:
            f.write("\n".join(self.contents_list))
        output = subprocess.check_output(["sbatch", file_name])
        stdout = output.decode("utf-8")
        print(stdout)
        self.id = stdout.split()[-1]
        os.system(f"rm {file_name}")
        return self.id

    def add_job_title(self, job_title):
        self.sbatch_name = job_title

    def execute_job(self):
        if self.sbatch_name == "":
            file_name = f"sbatch_{secrets.token_hex(16)}"
        else:
            file_name = f"{self.sbatch_name}_{secrets.token_hex(16)}.sh"

        with open(f"{file_name}.sh", "w") as f:
            f.write("\n".join(self.contents_list))
        os.system(f"bash {file_name}")
        os.system(f"rm {file_name}")