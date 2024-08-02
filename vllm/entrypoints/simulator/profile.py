import json
from collections import defaultdict


class SimulationProfile:
    """Profiling data structure capturing the timing of a single forward pass."""

    def __init__(self):
        self.prefill_timing = defaultdict(list)
        self.decode_timing = defaultdict(list)

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
