from vllm.entrypoints.simulator import vLLMSimulator, WorkloadRequest

workloads = [
    WorkloadRequest(0.0, 10, 10),
    WorkloadRequest(0.0, 20, 20),
    WorkloadRequest(0.0, 10, 10),
    WorkloadRequest(0.0, 20, 20),
    WorkloadRequest(0.05, 30, 30),
    WorkloadRequest(0.1, 40, 40),
    WorkloadRequest(0.2, 50, 50),
]

engine = vLLMSimulator(model="facebook/opt-125m")
engine.profile(workloads)
