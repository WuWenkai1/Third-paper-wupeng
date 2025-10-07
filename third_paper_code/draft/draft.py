import numpy as np
import random
from dataclasses import dataclass

@dataclass
class JobInstance:
    r: np.ndarray  # release times
    p: np.ndarray  # processing times
    d: np.ndarray  # due dates

def sample_instance(n=10, r_low=0, r_high=30, p_low=0, p_high=5, extra_due_high=20, seed=None) -> JobInstance:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    r = np.random.uniform(r_low, r_high, size=n).round(0).astype(np.float32)
    p = np.random.uniform(p_low, p_high, size=n).round(0).astype(np.float32)
    slack = np.random.uniform(0, extra_due_high, size=n).astype(np.float32)
    d = (r + p + slack).astype(np.float32)
    return JobInstance(r=r, p=p, d=d)

if __name__ == "__main__":
    inst = sample_instance(n=10, seed=2)
    print("r:", inst.r)
    print("p:", inst.p)
    print("d:", inst.d)