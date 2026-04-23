from __future__ import annotations

import os
from json import loads
from pathlib import Path
from time import time_ns
from urllib import request

import pytest

from greencompute.client import GreenferenceClient


@pytest.mark.skipif(
    not os.getenv("GREENFERENCE_LIVE_STACK") or not os.getenv("GREENFERENCE_API_KEY"),
    reason="requires GREENFERENCE_LIVE_STACK=1 and GREENFERENCE_API_KEY",
)
def test_live_stack_build_deploy_and_invoke(tmp_path: Path) -> None:
    base_url = os.getenv("GREENFERENCE_API_URL", "http://127.0.0.1:8000")
    control_plane_url = os.getenv("GREENFERENCE_CONTROL_PLANE_URL", "http://127.0.0.1:28001")
    miner_url = os.getenv("GREENFERENCE_MINER_URL", "http://127.0.0.1:28004")
    failover_miner_url = os.getenv("GREENFERENCE_FAILOVER_MINER_URL", "http://127.0.0.1:28005")
    api_key = os.environ["GREENFERENCE_API_KEY"]
    admin_api_key = os.getenv("GREENFERENCE_ADMIN_API_KEY")
    client = GreenferenceClient(base_url=base_url, api_key=api_key, timeout_seconds=60.0, max_retries=1)

    if admin_api_key:
        def _admin_request_json(method: str, url: str) -> list[dict] | dict:
            req = request.Request(url=url, method=method)
            req.add_header("X-API-Key", admin_api_key)
            with request.urlopen(req) as response:  # noqa: S310
                return loads(response.read().decode())

        deployments = _admin_request_json("GET", f"{control_plane_url}/platform/v1/debug/deployments")
        assert isinstance(deployments, list)
        for deployment in deployments:
            if deployment["state"] not in {"scheduled", "pulling", "starting", "ready"}:
                continue
            if deployment.get("hotkey") == "miner-local":
                request.urlopen(  # noqa: S310
                    request.Request(
                        url=f"{miner_url}/agent/v1/deployments/{deployment['deployment_id']}/terminate",
                        method="POST",
                    )
                ).read()
            elif deployment.get("hotkey") == "miner-failover":
                request.urlopen(  # noqa: S310
                    request.Request(
                        url=f"{failover_miner_url}/agent/v1/deployments/{deployment['deployment_id']}/terminate",
                        method="POST",
                    )
                ).read()
            _admin_request_json(
                "POST",
                f"{control_plane_url}/platform/v1/debug/deployments/{deployment['deployment_id']}/cleanup",
            )

    project = tmp_path / "live_sdk"
    project.mkdir()
    suffix = str(time_ns())
    (project / "app.py").write_text(
        f"""
from greencompute import Image, NodeSelector, Workload

image = (
    Image(username="demo", name="live-stack", tag="latest")
    .from_base("python:3.12-slim")
    .run_command("echo live stack")
)

workload = Workload(
    name="live-stack-demo-{suffix}",
    image=image,
    node_selector=NodeSelector(gpu_count=1),
    model_identifier="demo/live-stack",
    workload_alias="live-stack-demo-{suffix}",
)
""",
        encoding="utf-8",
    )

    old_cwd = Path.cwd()
    try:
        os.chdir(project)
        from greencompute.loader import load_workload
        from greencompute.packaging import package_workload

        loaded = load_workload(f"{project / 'app.py'}:workload")
        packaged = package_workload(loaded.module_path, loaded.workload)
        uploaded = client.upload_build_context(
            {
                "context_archive_b64": packaged.archive_b64,
                "context_archive_name": packaged.archive_name,
            }
        )
        build = client.build(loaded.workload.to_build_payload(context_uri=uploaded["context_uri"]))
        build_info = client.wait_for_build_info(build["build_id"], timeout_seconds=600.0, poll_interval_seconds=2.0)
        assert build_info.status == "published"

        workload = client.create_workload(loaded.workload.to_workload_payload())
        deployment = client.deploy(
            {
                "workload_id": workload["workload_id"],
                "requested_instances": 1,
                "accept_fee": True,
            }
        )
        deployment_info = client.wait_for_deployment_info(
            deployment["deployment_id"],
            timeout_seconds=900.0,
            poll_interval_seconds=5.0,
        )
        assert deployment_info.state == "ready"

        response = client.invoke_workload(loaded.workload.invocation_model, message="hello")
        assert isinstance(response, dict)
        assert response
    finally:
        os.chdir(old_cwd)
