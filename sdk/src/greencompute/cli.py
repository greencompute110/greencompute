"""GreenCompute CLI."""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from greencompute.client import (
    GreenComputeClient,
    GreenComputeConnectionError,
    GreenComputeHTTPError,
    GreenComputeTimeoutError,
)
from greencompute.config import default_config_path, get_config, init_config, mask_secret, save_config, unset_config
from greencompute.loader import load_workload
from greencompute.packaging import package_workload

app = typer.Typer(no_args_is_help=True, help="GreenCompute SDK and CLI")
console = Console()


def _client(
    ctx: typer.Context | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> GreenComputeClient:
    config = get_config()
    obj = ctx.obj if ctx and hasattr(ctx, "obj") and ctx.obj else {}
    return GreenComputeClient(
        base_url=base_url or obj.get("base_url") or config.api_base_url,
        api_key=api_key or obj.get("api_key") or config.api_key,
    )


def _emit(data: Any) -> None:
    console.print_json(json.dumps(data, default=str, indent=2))


def _render_build_log(entry: dict) -> str:
    if "message" in entry and "stage" in entry:
        return f"[{entry['stage']}] {entry['message']}"
    if entry.get("event") == "end":
        return f"[build] terminal status={entry.get('status')}"
    if "status" in entry:
        return f"[build] status={entry['status']}"
    return json.dumps(entry, default=str)


def _render_build_table(items: list[dict], *, title: str = "Builds") -> None:
    if not items:
        console.print("No builds found.")
        return
    table = Table(title=title)
    table.add_column("Build ID", style="cyan")
    table.add_column("Image")
    table.add_column("Status")
    table.add_column("Executor")
    for build in items:
        table.add_row(
            build.get("build_id", ""),
            build.get("image", ""),
            build.get("status", ""),
            build.get("executor_name", "") or "",
        )
    console.print(table)


def _render_deployment_table(items: list[dict]) -> None:
    if not items:
        console.print("No deployments found.")
        return
    table = Table(title="Deployments")
    table.add_column("Deployment ID", style="cyan")
    table.add_column("Workload ID")
    table.add_column("State")
    table.add_column("Instances")
    table.add_column("Endpoint")
    table.add_column("Fee Ack")
    for deployment in items:
        table.add_row(
            deployment.get("deployment_id", ""),
            deployment.get("workload_id", ""),
            deployment.get("state", ""),
            str(deployment.get("requested_instances", "")),
            deployment.get("endpoint", "") or "",
            "" if deployment.get("fee_acknowledged") is None else ("yes" if deployment.get("fee_acknowledged") else "no"),
        )
    console.print(table)


def _load_packaged(module_ref: str):
    loaded = load_workload(module_ref)
    packaged = package_workload(loaded.module_path, loaded.workload)
    return loaded, packaged


def _ensure_built(
    client: GreenComputeClient,
    module_ref: str,
    *,
    public: bool = False,
    wait: bool = False,
) -> dict:
    loaded, packaged = _load_packaged(module_ref)
    image_ref = loaded.workload.image_ref
    history = client.list_image_history(image_ref)
    published = next((item for item in history if item.get("status") == "published"), None)
    if published is not None:
        return published
    staged = client.upload_build_context(
        {
            "context_archive_b64": packaged.archive_b64,
            "context_archive_name": packaged.archive_name,
        }
    )
    build = client.build(
        loaded.workload.to_build_payload(
            context_uri=staged["context_uri"],
            public=loaded.workload.public or public,
        )
    )
    if wait:
        for entry in client.stream_build_logs(build["build_id"], follow=True):
            console.print(_render_build_log(entry))
    return client.wait_for_build(build["build_id"])


def _render_workload_table(items: list[dict]) -> None:
    if not items:
        console.print("No workloads found.")
        return
    table = Table(title="Workloads")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Image")
    table.add_column("Kind")
    table.add_column("Alias")
    table.add_column("Public")
    for workload in items:
        table.add_row(
            workload.get("workload_id", ""),
            workload.get("name", ""),
            workload.get("image", ""),
            workload.get("kind", ""),
            workload.get("workload_alias", "") or "",
            "yes" if workload.get("public") else "no",
        )
    console.print(table)


@app.callback()
def main_callback(
    ctx: typer.Context,
    base_url: str | None = typer.Option(None, "--base-url", envvar="GREENCOMPUTE_API_URL"),
    api_key: str | None = typer.Option(None, "--api-key", envvar="GREENCOMPUTE_API_KEY"),
) -> None:
    """GreenCompute CLI - manage workloads, images, deployments, and inference."""
    ctx.obj = {"base_url": base_url, "api_key": api_key}


config_app = typer.Typer(help="Manage local SDK configuration")
app.add_typer(config_app, name="config")


@config_app.command("show", help="Show the resolved SDK configuration")
def config_show() -> None:
    config = get_config()
    _emit(
        {
            "config_path": str(default_config_path()),
            "base_url": config.api_base_url,
            "api_key": mask_secret(config.api_key),
        }
    )


@config_app.command("init", help="Initialize SDK configuration on disk")
def config_init(
    base_url: str = typer.Option("http://127.0.0.1:8000", help="Default GreenCompute API URL"),
    api_key: str | None = typer.Option(None, help="Default API key"),
) -> None:
    saved = init_config(api_base_url=base_url, api_key=api_key)
    _emit(
        {
            "config_path": str(default_config_path()),
            "base_url": saved.api_base_url,
            "api_key": mask_secret(saved.api_key),
        }
    )


@config_app.command("set", help="Persist SDK configuration to disk")
def config_set(
    base_url: str | None = typer.Option(None, help="Default GreenCompute API URL"),
    api_key: str | None = typer.Option(None, help="Default API key"),
) -> None:
    saved = save_config(api_base_url=base_url, api_key=api_key)
    _emit(
        {
            "config_path": str(default_config_path()),
            "base_url": saved.api_base_url,
            "api_key": mask_secret(saved.api_key),
        }
    )


@config_app.command("unset", help="Remove stored config values")
def config_unset(
    base_url: bool = typer.Option(False, help="Unset stored base URL"),
    api_key: bool = typer.Option(False, help="Unset stored API key"),
) -> None:
    saved = unset_config(api_base_url=base_url, api_key=api_key)
    _emit(
        {
            "config_path": str(default_config_path()),
            "base_url": saved.api_base_url,
            "api_key": mask_secret(saved.api_key),
        }
    )


# --- Register (no auth) ---
@app.command(help="Create an account")
def register(
    ctx: typer.Context,
    username: str = typer.Option(..., help="Username"),
    email: str | None = typer.Option(None, help="Email"),
) -> None:
    client = _client(ctx)
    _emit(client.register({"username": username, "email": email}))


# --- Workloads ---
workloads_app = typer.Typer(help="Manage workloads")
app.add_typer(workloads_app, name="workloads")


@workloads_app.command("list", help="List workloads")
def workloads_list(ctx: typer.Context) -> None:
    client = _client(ctx)
    _render_workload_table(client.list_workloads())


@workloads_app.command("create", help="Create a workload from a Python module ref")
def workloads_create(
    ctx: typer.Context,
    module_ref: str = typer.Argument(..., help="Python ref like path/to/file.py:workload"),
    public: bool | None = typer.Option(None, "--public/--private", help="Visibility override"),
) -> None:
    client = _client(ctx)
    loaded = load_workload(module_ref)
    payload = loaded.workload.to_workload_payload()
    if public is not None:
        payload["public"] = public
    _emit(client.create_workload(payload))


@workloads_app.command("get", help="Get a workload by ID")
def workloads_get(
    ctx: typer.Context,
    workload_id: str = typer.Argument(..., help="Workload ID"),
) -> None:
    client = _client(ctx)
    _emit(client.get_workload(workload_id))


@workloads_app.command("delete", help="Delete a workload")
def workloads_delete(
    ctx: typer.Context,
    workload_id: str = typer.Argument(..., help="Workload ID"),
) -> None:
    client = _client(ctx)
    _emit(client.delete_workload(workload_id))


@workloads_app.command("update", help="Update workload metadata and lifecycle policy")
def workloads_update(
    ctx: typer.Context,
    workload_id: str = typer.Argument(..., help="Workload ID"),
    display_name: str | None = typer.Option(None, help="Display name"),
    readme: str | None = typer.Option(None, help="Workload README"),
    logo_uri: str | None = typer.Option(None, help="Logo URI"),
    tag: list[str] | None = typer.Option(None, "--tag", help="Tag to apply, repeatable"),
    clear_tags: bool = typer.Option(False, help="Clear all tags"),
    public: bool | None = typer.Option(None, "--public/--private", help="Visibility override"),
    workload_alias: str | None = typer.Option(None, help="Alias used for invocation routing"),
    clear_workload_alias: bool = typer.Option(False, help="Clear workload alias"),
    ingress_host: str | None = typer.Option(None, help="Ingress host"),
    pricing_class: str | None = typer.Option(None, help="Pricing class"),
    scaling_threshold: float | None = typer.Option(None, help="Scale-up threshold"),
    shutdown_after_seconds: int | None = typer.Option(None, help="Idle shutdown timeout"),
    warmup_enabled: bool | None = typer.Option(None, "--warmup-enabled/--warmup-disabled", help="Warmup toggle"),
    warmup_path: str | None = typer.Option(None, help="Warmup path"),
) -> None:
    client = _client(ctx)
    payload: dict[str, Any] = {}
    if display_name is not None:
        payload["display_name"] = display_name
    if readme is not None:
        payload["readme"] = readme
    if logo_uri is not None:
        payload["logo_uri"] = logo_uri
    if clear_tags:
        payload["tags"] = []
    elif tag is not None:
        payload["tags"] = tag
    if public is not None:
        payload["public"] = public
    if clear_workload_alias:
        payload["clear_workload_alias"] = True
    elif workload_alias is not None:
        payload["workload_alias"] = workload_alias
    if ingress_host is not None:
        payload["ingress_host"] = ingress_host
    if pricing_class is not None:
        payload["pricing_class"] = pricing_class
    lifecycle: dict[str, Any] = {}
    if scaling_threshold is not None:
        lifecycle["scaling_threshold"] = scaling_threshold
    if shutdown_after_seconds is not None:
        lifecycle["shutdown_after_seconds"] = shutdown_after_seconds
    if warmup_enabled is not None:
        lifecycle["warmup_enabled"] = warmup_enabled
    if warmup_path is not None:
        lifecycle["warmup_path"] = warmup_path
    if lifecycle:
        payload["lifecycle"] = lifecycle
    _emit(client.update_workload(workload_id, payload))


@workloads_app.command("share", help="Share a workload with another user")
def workloads_share(
    ctx: typer.Context,
    workload_id: str = typer.Argument(..., help="Workload ID"),
    user_id: str = typer.Option(..., "--user-id", help="User ID to share with"),
    permission: str = typer.Option("invoke", help="Permission name"),
) -> None:
    client = _client(ctx)
    _emit(
        client.share_workload(
            workload_id,
            {"shared_with_user_id": user_id, "permission": permission},
        )
    )


@workloads_app.command("shares", help="List workload shares")
def workloads_shares(
    ctx: typer.Context,
    workload_id: str = typer.Argument(..., help="Workload ID"),
) -> None:
    client = _client(ctx)
    _emit(client.list_workload_shares(workload_id))


@workloads_app.command("warmup", help="Warm up a workload and stream events")
def workloads_warmup(
    ctx: typer.Context,
    workload_id: str = typer.Argument(..., help="Workload ID"),
) -> None:
    client = _client(ctx)
    for event in client.workload_warmup(workload_id):
        _emit(event)


@workloads_app.command("utilization", help="Show workload utilization")
def workloads_utilization(
    ctx: typer.Context,
    workload_id: str = typer.Argument(..., help="Workload ID"),
) -> None:
    client = _client(ctx)
    _emit(client.get_workload_utilization(workload_id))


@workloads_app.command("create-vllm", help="Create a VLLM workload from HuggingFace model")
def workloads_create_vllm(
    ctx: typer.Context,
    model: str = typer.Option(..., help="HuggingFace model (org/model)"),
    username: str = typer.Option("greencompute", help="Image owner/namespace"),
    name: str | None = typer.Option(None, help="Workload name"),
    display_name: str | None = typer.Option(None, help="Display name"),
    workload_alias: str | None = typer.Option(None, help="Alias used for invocation routing"),
) -> None:
    from greencompute.templates import build_vllm_workload

    client = _client(ctx)
    workload = build_vllm_workload(
        username=username,
        name=name or model.rsplit("/", 1)[-1].replace(".", "-"),
        model_identifier=model,
        display_name=display_name,
        workload_alias=workload_alias,
    )
    _emit(client.create_workload(workload.workload.to_workload_payload()))


@workloads_app.command("create-diffusion", help="Create a diffusion workload")
def workloads_create_diffusion(
    ctx: typer.Context,
    model: str = typer.Option(..., help="Model identifier"),
    username: str = typer.Option("greencompute", help="Image owner/namespace"),
    name: str = typer.Option(..., help="Workload name"),
    display_name: str | None = typer.Option(None, help="Display name"),
    workload_alias: str | None = typer.Option(None, help="Alias used for invocation routing"),
) -> None:
    from greencompute.templates import build_diffusion_workload

    client = _client(ctx)
    workload = build_diffusion_workload(
        username=username,
        name=name,
        model_identifier=model,
        display_name=display_name,
        workload_alias=workload_alias,
    )
    _emit(client.create_workload(workload.workload.to_workload_payload()))


@workloads_app.command("guess", help="Guess GPU requirements for a HuggingFace model")
def workloads_guess(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="HuggingFace model (org/model)"),
) -> None:
    client = _client(ctx)
    _emit(client.guess_vllm_config(model))


# --- Images ---
images_app = typer.Typer(help="Manage images")
app.add_typer(images_app, name="images")


@images_app.command("list", help="List images")
def images_list(ctx: typer.Context) -> None:
    client = _client(ctx)
    items = client.list_images()
    if not items:
        console.print("No images found.")
        return
    table = Table(title="Images")
    table.add_column("Build ID", style="cyan")
    table.add_column("Image")
    table.add_column("Status")
    for b in items:
        table.add_row(
            b.get("build_id", ""),
            b.get("image", ""),
            b.get("status", ""),
        )
    console.print(table)


@images_app.command("get", help="Get a build by ID")
def images_get(
    ctx: typer.Context,
    build_id: str = typer.Argument(..., help="Build ID"),
) -> None:
    client = _client(ctx)
    _emit(client.get_build(build_id))


@images_app.command("history", help="Show build history for an image")
def images_history(
    ctx: typer.Context,
    image: str = typer.Argument(..., help="Image reference"),
) -> None:
    client = _client(ctx)
    _render_build_table(client.list_image_history(image), title=f"Image History: {image}")


builds_app = typer.Typer(help="Manage builds")
app.add_typer(builds_app, name="builds")


@builds_app.command("list", help="List builds")
def builds_list(ctx: typer.Context) -> None:
    client = _client(ctx)
    _render_build_table(client.list_builds())


@builds_app.command("get", help="Get a build by ID")
def builds_get(
    ctx: typer.Context,
    build_id: str = typer.Argument(..., help="Build ID"),
) -> None:
    client = _client(ctx)
    _emit(client.get_build(build_id))


@builds_app.command("logs", help="Stream build logs")
def builds_logs(
    ctx: typer.Context,
    build_id: str = typer.Argument(..., help="Build ID"),
    follow: bool = typer.Option(False, help="Follow until terminal build state"),
) -> None:
    client = _client(ctx)
    for entry in client.stream_build_logs(build_id, follow=follow):
        console.print(_render_build_log(entry))


@builds_app.command("wait", help="Wait for a build to reach a terminal state")
def builds_wait(
    ctx: typer.Context,
    build_id: str = typer.Argument(..., help="Build ID"),
    timeout: float = typer.Option(300.0, help="Timeout in seconds"),
    poll_interval: float = typer.Option(1.0, help="Polling interval in seconds"),
) -> None:
    client = _client(ctx)
    build = client.wait_for_build(
        build_id,
        timeout_seconds=timeout,
        poll_interval_seconds=poll_interval,
    )
    _emit(build)
    if build.get("status") != "published":
        raise typer.Exit(code=1)


# --- API Keys ---
keys_app = typer.Typer(help="Manage API keys")
app.add_typer(keys_app, name="keys")


@keys_app.command("create", help="Create an API key")
def keys_create(
    ctx: typer.Context,
    name: str = typer.Option(..., help="Key name"),
    user_id: str | None = typer.Option(None, help="User ID (admin only)"),
    admin: bool = typer.Option(False, help="Admin key"),
) -> None:
    client = _client(ctx)
    payload = {"name": name, "admin": admin}
    if user_id:
        payload["user_id"] = user_id
    _emit(client.create_api_key(payload))


@keys_app.command("list", help="List API keys")
def keys_list(ctx: typer.Context) -> None:
    client = _client(ctx)
    items = client.list_api_keys()
    if not items:
        console.print("No API keys found.")
        return
    table = Table(title="API Keys")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Admin")
    for k in items:
        table.add_row(
            k.get("key_id", ""),
            k.get("name", ""),
            "yes" if k.get("admin") else "no",
        )
    console.print(table)


@keys_app.command("get", help="Get an API key by ID")
def keys_get(
    ctx: typer.Context,
    key_id: str = typer.Argument(..., help="Key ID"),
) -> None:
    client = _client(ctx)
    _emit(client.get_api_key(key_id))


@keys_app.command("delete", help="Delete an API key")
def keys_delete(
    ctx: typer.Context,
    key_id: str = typer.Argument(..., help="Key ID"),
) -> None:
    client = _client(ctx)
    _emit(client.delete_api_key(key_id))


# --- Secrets ---
secrets_app = typer.Typer(help="Manage secrets")
app.add_typer(secrets_app, name="secrets")


@secrets_app.command("list", help="List secrets")
def secrets_list(ctx: typer.Context) -> None:
    client = _client(ctx)
    _emit(client.list_secrets())


@secrets_app.command("create", help="Create a secret")
def secrets_create(
    ctx: typer.Context,
    purpose: str = typer.Option(..., help="Purpose"),
    key: str = typer.Option(..., help="Key/value"),
) -> None:
    client = _client(ctx)
    _emit(client.create_secret({"purpose": purpose, "key": key}))


@secrets_app.command("delete", help="Delete a secret")
def secrets_delete(
    ctx: typer.Context,
    secret_id: str = typer.Argument(..., help="Secret ID"),
) -> None:
    client = _client(ctx)
    _emit(client.delete_secret(secret_id))


# --- Build ---
@app.command(help="Start an image build")
def build(
    ctx: typer.Context,
    module_ref: str | None = typer.Argument(None, help="Python ref like path/to/file.py:workload"),
    image: str | None = typer.Option(None, help="Image name"),
    context_uri: str | None = typer.Option(None, help="Context URI"),
    dockerfile_path: str = typer.Option("Dockerfile", help="Dockerfile path"),
    public: bool = typer.Option(False, help="Public image"),
    wait: bool = typer.Option(False, help="Wait for build completion"),
) -> None:
    client = _client(ctx)
    if module_ref is not None:
        loaded, packaged = _load_packaged(module_ref)
        console.print(
            f"Packaging {len(packaged.included_paths)} files, "
            f"excluding {len(packaged.excluded_paths)} paths, "
            f"archive={packaged.archive_size_bytes} bytes"
        )
        for path in packaged.included_paths[:10]:
            console.print(f" include: {path}")
        if packaged.excluded_paths:
            for path in packaged.excluded_paths[:5]:
                console.print(f" exclude: {path}")
        staged = client.upload_build_context(
            {
                "context_archive_b64": packaged.archive_b64,
                "context_archive_name": packaged.archive_name,
            }
        )
        result = client.build(
            loaded.workload.to_build_payload(
                context_uri=staged["context_uri"],
                public=loaded.workload.public or public,
            )
        )
        if wait:
            for entry in client.stream_build_logs(result["build_id"], follow=True):
                console.print(_render_build_log(entry))
            result = client.wait_for_build(result["build_id"])
            if result.get("status") != "published":
                _emit(result)
                raise typer.Exit(code=1)
        _emit(result)
        return
    if image is None or context_uri is None:
        raise typer.BadParameter("build requires <module_ref> or both --image and --context-uri")
    _emit(
        client.build(
            {
                "image": image,
                "context_uri": context_uri,
                "dockerfile_path": dockerfile_path,
                "public": public,
            }
        )
    )


# --- Deploy ---
@app.command(help="Deploy a workload")
def deploy(
    ctx: typer.Context,
    module_ref: str | None = typer.Argument(None, help="Python ref like path/to/file.py:workload"),
    workload_id: str | None = typer.Option(None, help="Workload ID"),
    name: str | None = typer.Option(None, help="Workload name (if creating new)"),
    image: str | None = typer.Option(None, help="Image (if creating new)"),
    gpu_count: int = typer.Option(1, help="GPU count"),
    min_vram_gb: int = typer.Option(16, help="Min VRAM per GPU"),
    requested_instances: int = typer.Option(1, help="Requested instances"),
    accept_fee: bool = typer.Option(False, help="Acknowledge and accept deployment fee"),
    public: bool = typer.Option(False, help="Build image as public if auto-building"),
    wait: bool = typer.Option(False, help="Wait for auto-build completion"),
) -> None:
    client = _client(ctx)
    if module_ref is not None:
        loaded = load_workload(module_ref)
        if not isinstance(loaded.workload.image, str):
            build = _ensure_built(client, module_ref, public=public, wait=wait)
            if build.get("status") != "published":
                console.print(f"image build did not publish successfully: {build.get('status')}")
                raise typer.Exit(code=1)
        workload = client.create_workload(loaded.workload.to_workload_payload())
        deployment = client.deploy(
            {
                "workload_id": workload["workload_id"],
                "requested_instances": requested_instances,
                "accept_fee": accept_fee,
            }
        )
        if wait:
            deployment = client.wait_for_deployment(deployment["deployment_id"])
            if str(deployment.get("state", "")).lower() != "ready":
                _emit(deployment)
                raise typer.Exit(code=1)
        _emit(deployment)
        return
    if workload_id is None:
        if not name or not image:
            raise typer.BadParameter("deploy requires --workload-id or both --name and --image")
        workload = client.create_workload(
            {
                "name": name,
                "image": image,
                "requirements": {
                    "gpu_count": gpu_count,
                    "min_vram_gb_per_gpu": min_vram_gb,
                },
            }
        )
        workload_id = workload["workload_id"]
    deployment = client.deploy(
        {
            "workload_id": workload_id,
            "requested_instances": requested_instances,
            "accept_fee": accept_fee,
        }
    )
    if wait:
        deployment = client.wait_for_deployment(deployment["deployment_id"])
        if str(deployment.get("state", "")).lower() != "ready":
            _emit(deployment)
            raise typer.Exit(code=1)
    _emit(deployment)


deployments_app = typer.Typer(help="Manage deployments")
app.add_typer(deployments_app, name="deployments")


@deployments_app.command("list", help="List deployments")
def deployments_list(ctx: typer.Context) -> None:
    client = _client(ctx)
    _render_deployment_table(client.list_deployments())


@deployments_app.command("get", help="Get a deployment by ID")
def deployments_get(
    ctx: typer.Context,
    deployment_id: str = typer.Argument(..., help="Deployment ID"),
) -> None:
    client = _client(ctx)
    _emit(client.get_deployment(deployment_id))


@deployments_app.command("update", help="Update a deployment")
def deployments_update(
    ctx: typer.Context,
    deployment_id: str = typer.Argument(..., help="Deployment ID"),
    requested_instances: int | None = typer.Option(None, help="Requested instances"),
    fee_acknowledged: bool | None = typer.Option(None, "--fee-acknowledged/--fee-unacknowledged", help="Deployment fee acknowledgment"),
) -> None:
    client = _client(ctx)
    payload: dict[str, Any] = {}
    if requested_instances is not None:
        payload["requested_instances"] = requested_instances
    if fee_acknowledged is not None:
        payload["fee_acknowledged"] = fee_acknowledged
    _emit(client.update_deployment(deployment_id, payload))


@deployments_app.command("wait", help="Wait for a deployment to reach a terminal or ready state")
def deployments_wait(
    ctx: typer.Context,
    deployment_id: str = typer.Argument(..., help="Deployment ID"),
    timeout: float = typer.Option(600.0, help="Timeout in seconds"),
    poll_interval: float = typer.Option(2.0, help="Polling interval in seconds"),
) -> None:
    client = _client(ctx)
    deployment = client.wait_for_deployment(
        deployment_id,
        timeout_seconds=timeout,
        poll_interval_seconds=poll_interval,
    )
    _emit(deployment)
    if str(deployment.get("state", "")).lower() != "ready":
        raise typer.Exit(code=1)


# --- Invoke ---
@app.command(help="Invoke a deployed workload defined by a Python module ref")
def run(
    ctx: typer.Context,
    module_ref: str = typer.Argument(..., help="Python ref like path/to/file.py:workload"),
    message: str = typer.Option(..., help="User message"),
    stream: bool = typer.Option(False, help="Stream response"),
) -> None:
    loaded = load_workload(module_ref)
    payload = {
        "model": loaded.workload.invocation_model,
        "messages": [{"role": "user", "content": message}],
    }
    client = _client(ctx)
    if stream:
        for line in client.invoke_workload(loaded.workload.invocation_model, message=message, stream=True):  # type: ignore[arg-type]
            print(line)
    else:
        _emit(client.invoke_workload(loaded.workload.invocation_model, message=message))


@app.command(help="Invoke chat completion")
def invoke(
    ctx: typer.Context,
    model: str = typer.Option(..., help="Model identifier"),
    message: str = typer.Option(..., help="User message"),
    stream: bool = typer.Option(False, help="Stream response"),
) -> None:
    client = _client(ctx)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
    }
    if stream:
        for line in client.invoke_stream(payload):
            print(line)
    else:
        _emit(client.invoke(payload))


def main() -> None:
    try:
        app()
    except GreenComputeHTTPError as exc:
        status = f"HTTP {exc.status_code}" if exc.status_code is not None else "HTTP error"
        detail = f": {exc.body}" if exc.body else ""
        console.print(f"{status}: {exc}{detail}", style="red")
        raise typer.Exit(code=1) from exc
    except GreenComputeTimeoutError as exc:
        console.print(f"Request timed out: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    except GreenComputeConnectionError as exc:
        console.print(f"Connection failed: {exc}", style="red")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    main()
