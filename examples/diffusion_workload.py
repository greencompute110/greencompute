from greencompute.templates import build_diffusion_workload

workload = build_diffusion_workload(
    username="demo",
    name="sdxl-diffusion",
    model_identifier="stabilityai/stable-diffusion-xl-base-1.0",
    display_name="SDXL Diffusion",
    workload_alias="sdxl-diffusion",
    tags=["diffusion", "image"],
).workload
