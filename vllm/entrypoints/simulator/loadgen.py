from dataclasses import dataclass


@dataclass
class WorkloadRequest:
    arrival_time: float
    num_input_tokens: int
    num_output_tokens: int

    # TODO: add fields for prefix sharing
