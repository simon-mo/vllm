from dataclasses import dataclass
import time
import json

import simpy

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.entrypoints.simulator.profile import SimulationProfile


class vLLMSimulator:

    def __init__(self, model="facebook/opt-125m") -> None:
        self.profile_engine = LLMEngine.from_engine_args(
            EngineArgs(model=model, enforce_eager=True))
        self.simulated_engine = LLMEngine.from_engine_args(
            EngineArgs(model=model, simulation_mode=True))

        self._reset()

    def _reset(self):
        self.env = simpy.rt.RealtimeEnvironment(factor=1, strict=True)
        self.generator_finished = False
        self.enqueued = simpy.Store(self.env, capacity=1)

    def _process_run_engine(self, engine, is_profile):
        start_time = time.time()
        while not self.generator_finished or self.engine.has_unfinished_requests(
        ):
            start = time.time()
            outputs = engine.step()
            if len(outputs) == 0:
                continue
            print("---")
            for output in outputs:
                output_metrics = {
                    "arrival_time":
                    output.metrics.arrival_time - start_time,
                    "last_token_time":
                    output.metrics.last_token_time - start_time,
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
            if is_profile:
                yield self.env.timeout(passed)
            else:
                if not self.engine.has_unfinished_requests():
                    yield self.enqueued.get()
                yield self.env.timeout(0.0006)  # fixed scheduler overhead

    def _process_request_generator(self, engine, workloads):
        curr_time = self.env.now
        # assume workloads are sorted by arrival time
        for i, workload in enumerate(workloads):
            if self.env.now != workload.arrival_time:
                assert self.env.now < workload.arrival_time
                yield self.env.timeout(workload.arrival_time - curr_time)

            engine.add_request(
                request_id=str(i),
                inputs=dict(prompt_token_ids=[0] * workload.num_input_tokens),
                params=SamplingParams(max_tokens=workload.num_output_tokens,
                                      ignore_eos=True),
            )

            if len(self.enqueued.items) == 0:
                # notify the engine that there is a new request
                self.enqueued.put(i)

        self.generator_finished = True
        if len(self.enqueued.items) == 0:
            # notify the engine that there is a new request
            self.enqueued.put(i)

    def profile(self, workloads):
        profile = SimulationProfile()
        self.profile_engine.model_executor.simulation_profile = profile

        self.env.process(self._process_request_generator(self.profile_engine, workloads))
        self.env.process(self._process_run_engine(self.profile_engine, is_profile=True))
        self.env.run()

        profile.save("profile.json")
    

    def profile_tokens_curve(self, workload_batch, n_trials=1):
        profile = SimulationProfile()
        self.profile_engine.model_executor.simulation_profile = profile
        idx = -1

        # warmup
        for batch in workload_batch:
            for workload in batch:
                idx += 1
                self.profile_engine.add_request(
                    request_id=str(idx),
                    inputs=dict(prompt_token_ids=[0] * workload.num_input_tokens),
                    params=SamplingParams(max_tokens=max(workload.num_output_tokens, 1),
                                            ignore_eos=True),
                )
            while self.profile_engine.has_unfinished_requests():
                self.profile_engine.step()

        # real run

        profile = SimulationProfile()
        self.profile_engine.model_executor.simulation_profile = profile


        for batch in workload_batch:
            for _ in range(n_trials):
                for workload in batch:
                    idx += 1
                    self.profile_engine.add_request(
                        request_id=str(idx),
                        inputs=dict(prompt_token_ids=[0] * workload.num_input_tokens),
                        params=SamplingParams(max_tokens=max(workload.num_output_tokens, 1),
                                            ignore_eos=True),
                    )
                while self.profile_engine.has_unfinished_requests():
                    self.profile_engine.step()
        
        return profile
    



# if __name__ == "__main__":
#     from vllm.utils import FlexibleArgumentParser
#     parser = FlexibleArgumentParser()
