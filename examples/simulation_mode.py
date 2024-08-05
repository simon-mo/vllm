from vllm.entrypoints.simulator import vLLMSimulator, WorkloadRequest

# workloads = [
#     WorkloadRequest(0.0, 10, 10),
#     WorkloadRequest(0.0, 20, 20),
#     WorkloadRequest(0.0, 10, 10),
#     WorkloadRequest(0.0, 20, 20),
#     WorkloadRequest(0.05, 30, 30),
#     WorkloadRequest(0.1, 40, 40),
#     WorkloadRequest(0.2, 50, 50),
# ]

# engine = vLLMSimulator(model="facebook/opt-125m")
# engine.profile(workloads)

# Profile workload batch with prefill 2->2048 tokens and decode 1->256 tokens, using power of 2 ranges, but with one mid point value in between.
# For decode workload, sample the input tokens to 64.

import numpy as np

prefill_sizes = np.logspace(np.log2(2), np.log2(8192), num=64,
                            base=2).round().astype(int)
decode_sizes = np.logspace(np.log2(2), np.log2(512), num=64,
                           base=2).round().astype(int)

workload_batch = []
# first measure prefill
for i in prefill_sizes:
    workload_batch.append([WorkloadRequest(0, i, 0)])
# then measure decode
for i in decode_sizes:
    # TODO: check ctx size > 1, and how it varies (it should not)
    # workload_batch.append([WorkloadRequest(0, 1, 2) for _ in range(i)])
    workload_batch.append([WorkloadRequest(0, 1, 2) for _ in range(i)])

print(prefill_sizes)
print(decode_sizes)

# workload_batch = [
#     [WorkloadRequest(0, 100, 0), WorkloadRequest(0, 100, 0)], # 200 prefil tokens
#     [WorkloadRequest(0, 200, 4), WorkloadRequest(0, 200, 4)], # 400 prefill tokens, 2 decode token per iter
# ]

# engine = vLLMSimulator(model="facebook/opt-125m")
# engine = vLLMSimulator(model="meta-llama/Meta-Llama-3-8B-Instruct")
engine = vLLMSimulator(model="meta-llama/Meta-Llama-3-70B-Instruct",
                       tensor_parallel_size=4)
profile = engine.profile_tokens_curve(workload_batch, n_trials=3)

# print(profile.prefill_timing) # this is a defaultdict(list)
# print(profile.decode_timing) # this is a defaultdict(list)

# turn this into a pandas dataframe

import pandas as pd

data = []
for k, v in profile.prefill_timing.items():
    for i in v:
        data.append({"size": k, "time_ms": i, "op": "prefill"})

for k, v in profile.decode_timing.items():
    for i in v:
        data.append({"size": k, "time_ms": i, "op": "decode"})

df = pd.DataFrame(data)

import sys

print("----")
df.to_csv(sys.stdout, index=False)
df.to_csv("profile-70b.csv", index=False)
