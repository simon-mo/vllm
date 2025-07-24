# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid

from openai import OpenAI

# 1: Set up your clients
openai_api_key = "EMPTY"
prefill_client = OpenAI(
    api_key=openai_api_key,
    base_url="http://IP_ADDRESS_OF_PREFILL_INSTANCE:8000/v1",
)
decode_client = OpenAI(
    api_key=openai_api_key,
    base_url="http://IP_ADDRESS_OF_DECODE_INSTANCE:8001/v1",
)
models = prefill_client.models.list()
model = models.data[0].id

# 2: Prefill
request_id = str(
    uuid.uuid4()
)  # maintain the same request_id for both prefill and decode
response = prefill_client.completions.create(
    model=model,
    # PD only works when the prompt is greater than vLLM's block size (16 tokens)
    prompt="A long prompt greater than 16 tokens",
    # Set this to force only prefill
    max_tokens=1,
    # This will be echoed back to pass to decode
    extra_body={
        "kv_transfer_params": {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
    },
    extra_headers={"X-Request-Id": request_id},
)
print("-" * 50)
print("Prefill results:")
print(response)


# 3: Decode
# Pass the kv_transfer_params from prefill to decode
decode_response = decode_client.completions.create(
    model=model,
    prompt="A robot may not injure a human being",
    extra_body={"kv_transfer_params": response.kv_transfer_params},
    extra_headers={"X-Request-Id": request_id},
)
print("-" * 50)
print("Decode results:")
print(decode_response)
print("-" * 50)
