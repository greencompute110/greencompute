from greencompute import Image, NodeSelector, Workload

image = (
    Image(username="demo", name="minimal-inference", tag="latest")
    .from_base("python:3.12-slim")
    .run_command("echo preparing minimal workload")
)

workload = Workload(
    name="minimal-inference",
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24),
    display_name="Minimal Inference",
    readme="A minimal GreenCompute inference workload defined in Python.",
    model_identifier="demo/minimal-model",
    workload_alias="minimal-inference",
)
