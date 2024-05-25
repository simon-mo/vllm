from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
import simpy
from dataclasses import dataclass
import time
import json


class SimulationProfile:
    """Profiling data structure capturing the timing of a single forward pass."""

    def __init__(self):
        self.prefill_timing = {}
        self.decode_timing = {}

    def record(self, prefill_tokens, decode_tokens, duration_ms):
        # TODO: use histogram, and sampling

        assert not all([prefill_tokens, decode_tokens]), "chunked prefill todo"

        if prefill_tokens:
            self.prefill_timing[str(prefill_tokens)] = duration_ms
        else:
            self.decode_timing[str(decode_tokens)] = duration_ms

    def get_estimate(self, prefill_tokens, decode_tokens):
        # TODO: sample

        assert not all([prefill_tokens, decode_tokens]), "chunked prefill todo"

        if prefill_tokens:
            return self.prefill_timing[str(prefill_tokens)]
        else:
            return self.decode_timing[str(decode_tokens)]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "prefill_timing": self.prefill_timing,
                    "decode_timing": self.decode_timing
                }, f)

    @classmethod
    def load(cls, path):
        o = cls()
        with open(path, "r") as f:
            data = json.load(f)
            o.prefill_timing = data["prefill_timing"]
            o.decode_timing = data["decode_timing"]
        return o


# workload characteristics
@dataclass
class WorkloadRequest:
    arrival_time: float
    num_input_tokens: int
    num_output_tokens: int

    # TODO: add fields for prefix sharing


workloads = [
    WorkloadRequest(0.0, 10, 10),
    WorkloadRequest(0.0, 20, 20),
    WorkloadRequest(0.0, 10, 10),
    WorkloadRequest(0.0, 20, 20),
    WorkloadRequest(0.05, 30, 30),
    WorkloadRequest(0.1, 40, 40),
    WorkloadRequest(0.2, 50, 50),
]

# SIMULATION_MODE = True
SIMULATION_MODE = False

engine = LLMEngine.from_engine_args(
    EngineArgs(model="facebook/opt-125m", simulation_mode=SIMULATION_MODE))
# env = simpy.Environment()
env = simpy.rt.RealtimeEnvironment(factor=1, strict=True)

if not SIMULATION_MODE:
    # enable profiling
    profile = SimulationProfile()
    engine.model_executor.simulation_profile = profile
else:
    profile = SimulationProfile.load("profile.json")

    engine.model_executor.simulation_profile = profile
    engine.model_executor.env = env

    time.time = lambda: env.now

generator_finished = False
enqueued = simpy.Store(env, capacity=1)


def request_generator(env):
    curr_time = env.now
    # assume workloads are sorted by arrival time
    for i, workload in enumerate(workloads):
        if env.now != workload.arrival_time:
            yield env.timeout(workload.arrival_time - curr_time)

        engine.add_request(
            request_id=str(i),
            prompt=None,
            prompt_token_ids=[0] * workload.num_input_tokens,
            params=SamplingParams(max_tokens=workload.num_output_tokens,
                                  ignore_eos=True),
        )

        if len(enqueued.items) == 0:
            # notify the engine that there is a new request
            enqueued.put(i)

    global generator_finished
    generator_finished = True
    if len(enqueued.items) == 0:
        # notify the engine that there is a new request
        enqueued.put(i)


def engine_runner(env):
    start_time = time.time()
    while not generator_finished or engine.has_unfinished_requests():
        start = time.time()
        outputs = engine.step()
        print("---")
        for output in outputs:
            output_metrics = {
                "arrival_time": output.metrics.arrival_time - start_time,
                "last_token_time": output.metrics.last_token_time - start_time,
                "first_scheduled_time":
                output.metrics.first_scheduled_time - start_time,
                "first_token_time":
                output.metrics.first_token_time - start_time
            }
            print(output.request_id, output_metrics)
        print("---")
        end = time.time()
        # TODO: use proper synchronization
        passed = end - start
        print(passed)
        if not SIMULATION_MODE:
            yield env.timeout(passed)
        else:
            if not engine.has_unfinished_requests():
                yield enqueued.get()
            yield env.timeout(0.0006)  # fixed scheduler overhead


env.process(request_generator(env))
env.process(engine_runner(env))
env.run()

if not SIMULATION_MODE:
    profile.save("profile.json")
