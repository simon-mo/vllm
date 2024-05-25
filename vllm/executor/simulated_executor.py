from typing import List, Set, Tuple, Optional

import simpy

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.sequence import CompletionSequenceGroupOutput, SequenceOutput, Logprob
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig, VisionLanguageConfig)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput

logger = init_logger(__name__)


class SimulatedExecutor(ExecutorBase):
    model_config: ModelConfig
    cache_config: CacheConfig
    parallel_config: ParallelConfig
    scheduler_config: SchedulerConfig
    device_config: DeviceConfig
    load_config: LoadConfig
    lora_config: Optional[LoRAConfig]
    vision_language_config: Optional[VisionLanguageConfig]
    speculative_config: Optional[SpeculativeConfig]

    env = simpy.Environment

    def _init_executor(self) -> None:
        pass

    def _init_worker(self):
        pass

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        # TODO: make it realistic
        return [int(1e5), int(1e5)]

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        pass

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        prefill_tokens = sum(
            seq_group.token_chunk_size
            for seq_group in execute_model_req.seq_group_metadata_list
            if seq_group.is_prompt)
        decode_tokens = sum(
            seq_group.token_chunk_size
            for seq_group in execute_model_req.seq_group_metadata_list
            if not seq_group.is_prompt)

        out = []
        ids = []
        for seq_group in execute_model_req.seq_group_metadata_list:
            ids.append(seq_group.request_id)

            for seq_id, _data in seq_group.seq_data.items():
                out.append(
                    CompletionSequenceGroupOutput(
                        samples=[
                            SequenceOutput(
                                parent_seq_id=seq_id,
                                output_token=0,
                                logprobs={
                                    0: Logprob(logprob=1),
                                },
                            ),
                        ],
                        prompt_logprobs=None,
                    ))
        # print("processed requests: ", ids)
        duration_ms = self.simulation_profile.get_estimate(
            prefill_tokens, decode_tokens)
        self.env.run(self.env.now + duration_ms / 1e3)
        return [SamplerOutput(outputs=out)]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError()

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError()

    def list_loras(self) -> Set[int]:
        raise NotImplementedError()

    def check_health(self) -> None:
        return


class SimulatedExecutorAsync(SimulatedExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        pass

    async def check_health_async(self) -> None:
        return
