from dataclasses import dataclass


@dataclass
class ComputeStats:
    flops: int
    reads: int
    writes: int
