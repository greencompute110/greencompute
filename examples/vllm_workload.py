from greencompute.templates import build_vllm_workload

workload = build_vllm_workload(
    username="demo",
    name="llama-vllm",
    model_identifier="meta-llama/Llama-3.2-1B-Instruct",
    display_name="Llama vLLM",
    workload_alias="llama-vllm",
    tags=["llm", "chat", "vllm"],
).workload
