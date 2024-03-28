import torch
from torch import nn
from transformers import PretrainedConfig
from typing import List, Optional, Tuple

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput, SequenceGroupOutput, SequenceOutput, Logprob

KVCache = Tuple[torch.Tensor, torch.Tensor]


class NoopModelForPerformanceBenchmark(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        return torch.zeros(1)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return torch.zeros(1)

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return SamplerOutput(outputs=[
            SequenceGroupOutput(
                samples=[
                    SequenceOutput(
                        parent_seq_id=sampling_metadata.seq_groups[0][0][0],
                        output_token=1,
                        logprobs={1: Logprob(logprob=1, decoded_token=None)})
                ],
                prompt_logprobs=None,
            )
        ])

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        pass
