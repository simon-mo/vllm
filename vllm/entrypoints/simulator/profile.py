import json
from collections import defaultdict
from contextlib import contextmanager
import time


@contextmanager
def profile_hook(execute_model_req, profile):
    prefill_tokens = sum(
        seq_group.token_chunk_size
        for seq_group in execute_model_req.seq_group_metadata_list
        if seq_group.is_prompt)
    decode_tokens = sum(
        seq_group.token_chunk_size
        for seq_group in execute_model_req.seq_group_metadata_list
        if not seq_group.is_prompt)

    start = time.perf_counter_ns()

    yield

    duration = (time.perf_counter_ns() - start) / 1e6

    print(
        f"prefill_tokens: {prefill_tokens}, decode_tokens: {decode_tokens}, duration_ms: {duration}"
    )
    profile.record(prefill_tokens, decode_tokens, duration)


class SimulationProfile:
    """Profiling data structure capturing the timing of a single forward pass."""

    def __init__(self):
        self.prefill_timing = defaultdict(list)
        self.decode_timing = defaultdict(list)

    def clear(self):
        self.prefill_timing.clear()
        self.decode_timing.clear()

    def record(self, prefill_tokens, decode_tokens, duration_ms):
        # TODO: use histogram, and sampling

        assert not all([prefill_tokens, decode_tokens]), "chunked prefill todo"

        if prefill_tokens:
            self.prefill_timing[str(prefill_tokens)].append(duration_ms)
        else:
            self.decode_timing[str(decode_tokens)].append(duration_ms)

    def get_estimate(self, prefill_tokens, decode_tokens):
        # TODO: sample

        assert not all([prefill_tokens, decode_tokens]), "chunked prefill todo"

        if prefill_tokens:
            return self.prefill_timing[str(prefill_tokens)][0]
        else:
            return self.decode_timing[str(decode_tokens)][0]

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
