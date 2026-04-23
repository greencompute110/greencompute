from greencompute import Image

image = (
    Image(username="demo", name="build-only", tag="latest")
    .from_base("python:3.12-slim")
    .apt_install("git", "curl")
    .run_command("pip install uv")
    .entrypoint("python", "-m", "http.server", "8000")
)
