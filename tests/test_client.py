import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from ai_diffusion import eventloop
from ai_diffusion.backend import resources
from ai_diffusion.backend.api import (
    CheckpointInput,
    ConditioningInput,
    ImageInput,
    LoraInput,
    SamplingInput,
    TaggerInput,
    WorkflowInput,
    WorkflowKind,
)
from ai_diffusion.backend.client import ClientEvent, ClientModels, TextOutput, resolve_arch
from ai_diffusion.backend.comfy_client import (
    ComfyClient,
    JobInfo,
    _extract_tagger_output,
    _find_anima_ip_adapter_resources,
    _find_anima_model_patch_resources,
    _find_ip_adapters,
    parse_url,
    websocket_url,
)
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo
from ai_diffusion.backend.network import NetworkError
from ai_diffusion.backend.resources import ControlMode, ResourceKind, resource_id
from ai_diffusion.backend.server import Server, ServerBackend, ServerState
from ai_diffusion.files import File, FileFormat, FileLibrary
from ai_diffusion.image import Extent
from ai_diffusion.platform_tools import get_cuda_devices
from ai_diffusion.style import Arch, Style
from ai_diffusion.util import ensure

from .config import default_checkpoint, server_dir
from .conftest import qtapp


@pytest.fixture(scope="session")
def comfy_server(qtapp):
    backend = ServerBackend.cpu
    if len(get_cuda_devices()) > 0:
        backend = ServerBackend.cuda

    server = Server(str(server_dir), backend)
    assert server.state is ServerState.stopped, (
        f"Expected server installation at {server_dir}. To create the default installation run"
        " `pytest tests/test_server.py --test-install`"
    )
    qtapp.run(server.start(port=8189))
    yield server
    qtapp.run(server.stop())


def make_default_work(size=512, steps=20):
    return WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput(default_checkpoint[Arch.sd15]),
        images=ImageInput.from_extent(Extent(size, size)),
        conditioning=ConditioningInput("a photo of a cat", "a photo of a dog"),
        sampling=SamplingInput("euler", "normal", cfg_scale=7.0, total_steps=steps),
    )


@qtapp
async def test_connect_bad_url(comfy_server):
    client = ComfyClient("bad_url")
    with pytest.raises(NetworkError):
        await client.connect()


@pytest.mark.parametrize("cancel_point", ["after_enqueue", "after_start", "after_sampling"])
@qtapp
async def test_cancel(comfy_server: Server, cancel_point):
    assert comfy_server.url is not None
    client = ComfyClient(comfy_server.url)
    await client.connect()
    async for _ in client.discover_models(refresh=False):
        pass
    job_id = None
    interrupted = False
    stage = 0

    async for msg in client.listen():
        if msg.event is ClientEvent.error:
            assert False, msg.error

        elif stage == 0:
            assert msg.event is not ClientEvent.finished
            assert msg.job_id in (job_id, "")
            if not job_id:
                job_id = await client.enqueue(make_default_work(steps=1000))
                assert client.queued_count == 1
            if not interrupted:
                if cancel_point == "after_enqueue":
                    await client.cancel([job_id])
                    interrupted = True
                if cancel_point == "after_start" and msg.event is ClientEvent.progress:
                    await client.interrupt()
                    interrupted = True
                if cancel_point == "after_sampling" and msg.progress > 0.1:
                    await client.interrupt()
                    interrupted = True
            if msg.event is ClientEvent.interrupted:
                assert msg.job_id == job_id
                assert not client.is_executing and client.queued_count == 0

                job_id = await client.enqueue(make_default_work(size=320, steps=1))
                stage = 1
                assert client.queued_count == 1
            elif msg.event is ClientEvent.progress:
                assert stage == 0

        elif stage == 1:
            assert msg.event is not ClientEvent.interrupted
            assert msg.job_id in (job_id, "")
            if msg.event is ClientEvent.finished:
                assert msg.images is not None and len(msg.images) > 0
                assert msg.images[0].extent == Extent(320, 320)
                break

    assert not client.is_executing and client.queued_count == 0


@qtapp
async def test_disconnect(comfy_server: Server):
    async def listen(client: ComfyClient):
        async for msg in client.listen():
            assert msg.event is ClientEvent.connected

    assert comfy_server.url is not None
    client = ComfyClient(comfy_server.url)
    await client.connect()
    task = eventloop._loop.create_task(listen(client))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not client.is_executing and client.queued_count == 0


@pytest.mark.parametrize(
    "url,expected_http,expected_ws",
    [
        ("http://localhost:8000", "http://localhost:8000", "ws://localhost:8000"),
        ("http://localhost:8000/", "http://localhost:8000", "ws://localhost:8000"),
        ("http://localhost:8000/foo", "http://localhost:8000/foo", "ws://localhost:8000/foo"),
        ("http://127.0.0.1:1234", "http://127.0.0.1:1234", "ws://127.0.0.1:1234"),
        ("localhost:8000", "http://localhost:8000", "ws://localhost:8000"),
        ("https://localhost:8000", "https://localhost:8000", "wss://localhost:8000"),
    ],
)
def test_parse_url(url, expected_http, expected_ws):
    parsed = parse_url(url)
    assert parsed == expected_http and websocket_url(parsed) == expected_ws


def _node_options(name: str, values: list[str]):
    return {"input": {"required": {name: [values]}}}


def test_anima_model_patch_discovery_is_node_gated():
    filenames = [
        "anima-lllite-inpainting-v2.safetensors",
        "anima-lllite-any-test-like-v2.safetensors",
        "anima-lllite-scribble-1.safetensors",
        "anima-lllite-lineart-1.safetensors",
        "anima-lllite-depth-1.safetensors",
        "anima-lllite-pose-1.safetensors",
    ]
    nodes = ComfyObjectInfo({
        "ModelPatchLoader": _node_options("name", filenames),
        "AnimaLLLiteApply": {},
    })
    discovered = _find_anima_model_patch_resources(nodes)
    assert discovered == {
        resource_id(ResourceKind.model_patch, Arch.anima, ControlMode.inpaint): filenames[0],
        resource_id(ResourceKind.model_patch, Arch.anima, ControlMode.universal): filenames[1],
        resource_id(ResourceKind.model_patch, Arch.anima, ControlMode.scribble): filenames[2],
        resource_id(ResourceKind.model_patch, Arch.anima, ControlMode.line_art): filenames[3],
        resource_id(ResourceKind.model_patch, Arch.anima, ControlMode.depth): filenames[4],
        resource_id(ResourceKind.model_patch, Arch.anima, ControlMode.pose): filenames[5],
    }

    assert not _find_anima_model_patch_resources(
        ComfyObjectInfo({"ModelPatchLoader": _node_options("name", filenames)})
    )
    assert not _find_anima_model_patch_resources(ComfyObjectInfo({"AnimaLLLiteApply": {}}))


def test_anima_ip_adapter_discovery_is_node_gated_and_exact():
    filename = "ip_adapter-Character_Reference-10.safetensors"
    id = resource_id(ResourceKind.ip_adapter, Arch.anima, ControlMode.reference)
    nodes = ComfyObjectInfo({
        "AnimaIPAdapterLoader": _node_options("ip_adapter_name", [filename]),
        "AnimaIPAdapterApply": {},
    })
    assert _find_anima_ip_adapter_resources(nodes) == {id: filename}

    incompatible = "ip_adapter-Character_Reference-10-old.safetensors"
    nodes.nodes["AnimaIPAdapterLoader"] = _node_options("ip_adapter_name", [incompatible])
    assert _find_anima_ip_adapter_resources(nodes) == {id: None}

    assert not _find_anima_ip_adapter_resources(
        ComfyObjectInfo({"AnimaIPAdapterLoader": _node_options("ip_adapter_name", [filename])})
    )
    assert not _find_anima_ip_adapter_resources(
        ComfyObjectInfo({"IPAdapterModelLoader": _node_options("ipadapter_file", [filename])})
    )
    assert id not in _find_ip_adapters([filename])


def test_anima_adapter_availability_is_reference_only():
    models = ClientModels()
    anima = models.for_arch(Arch.anima)
    assert anima.find_control(ControlMode.reference) is None

    filename = "ip_adapter-Character_Reference-10.safetensors"
    models.resources[resource_id(ResourceKind.ip_adapter, Arch.anima, ControlMode.reference)] = (
        filename
    )
    assert anima.find_control(ControlMode.reference) == filename
    assert anima.find_control(ControlMode.face) is None
    assert anima.find_control(ControlMode.style) is None
    assert anima.find_control(ControlMode.composition) is None


def check_client_info(client: ComfyClient):
    assert client.device_info.type in ["cpu", "cuda"]
    assert client.device_info.name != ""
    assert client.device_info.vram > 0

    assert len(client.models.checkpoints) > 0
    for filename, cp in client.models.checkpoints.items():
        assert cp.filename == filename
        assert cp.filename.startswith(cp.name)
        assert cp.format is FileFormat.checkpoint

    assert len(client.models.resources) >= len(resources.required_resource_ids)
    inpaint = client.models.for_arch(Arch.sd15).control[ControlMode.inpaint]
    assert inpaint and "inpaint" in inpaint


def check_resolve_sd_version(client: ComfyClient, arch: Arch):
    checkpoint = next(cp for cp in client.models.checkpoints.values() if cp.arch == arch)
    style = Style(Path("dummy"))
    style.architecture = Arch.auto
    style.checkpoints = [checkpoint.filename]
    assert resolve_arch(style, client) == arch
    assert resolve_arch(style, None) == arch


@qtapp
async def test_info(pytestconfig, comfy_server: Server):
    assert comfy_server.url is not None
    client = ComfyClient(comfy_server.url)
    await client.connect()
    async for _ in client.discover_models(refresh=False):
        pass
    check_client_info(client)
    await client.refresh()
    check_client_info(client)
    check_resolve_sd_version(client, Arch.sd15)
    # check_resolve_sd_version(client, Arch.sdxl) # no SDXL checkpoint in default installation


@qtapp
async def test_upload_lora(comfy_server: Server, tmp_path: Path):
    lora_path = tmp_path / "test-lora.safetensors"
    lora_path.write_bytes(b"testdata" * 1024 * 1024)

    files = FileLibrary.instance()
    file = files.loras.add(File.local(lora_path, compute_hash=True))

    assert comfy_server.url is not None
    client = ComfyClient(comfy_server.url)
    await client.connect()
    if file.id in client.models.loras:
        client.models.loras.remove(file.id)

    input = make_default_work()
    assert input.models is not None
    input.models.loras = [LoraInput(file.id, 1.0, storage_id=ensure(file.hash))]

    task = asyncio.get_running_loop().create_task(client.upload_loras(input, "JOB-ID"))
    upload_progress = 0
    async for msg in client.listen():
        if msg.event is ClientEvent.upload:
            assert msg.job_id == "JOB-ID"
            assert msg.progress >= upload_progress
            upload_progress = msg.progress
            if upload_progress == 1.0:
                break

    await task
    assert file.id in client.models.loras


def test_tag_output_extraction():
    for payload, expected in [
        (["1girl, solo"], "1girl, solo"),
        ("landscape, sky", "landscape, sky"),
        ([""], ""),
    ]:
        msg = {"data": {"node": "2", "output": {"tags": payload}}}
        output = _extract_tagger_output("job", msg)
        assert output is not None
        assert output.event is ClientEvent.output
        assert output.job_id == "job"
        assert output.result == TextOutput("2", "Tags", expected, "text/plain")


def _tag_work():
    return WorkflowInput(WorkflowKind.tag, tagger=TaggerInput("wd-v1-4-moat-tagger-v2"))


def _event(type: str, job_id: str, **data):
    return json.dumps({"type": type, "data": {"prompt_id": job_id, **data}})


def _client_messages(client: ComfyClient):
    messages = []
    while not client._messages.empty():
        messages.append(client._messages.get_nowait())
    return messages


@qtapp
async def test_tag_only_websocket_completion():
    client = ComfyClient("http://mock")
    first = JobInfo("tag-1", _tag_work(), node_count=2)
    second = JobInfo("tag-2", _tag_work(), node_count=2)
    client._waiting_job.set(first)

    async def websocket():
        yield _event("execution_start", first.id)
        yield _event(
            "executed",
            first.id,
            node="2",
            output={"tags": ["1girl, solo"]},
        )
        yield _event("executing", first.id, node=None)
        client._waiting_job.set(second)
        yield _event("execution_start", second.id)
        yield _event("execution_cached", second.id, nodes=["1", "2"])
        yield _event("executing", second.id, node=None)

    await client._listen_websocket(cast(Any, websocket()))
    messages = _client_messages(client)
    assert [
        (msg.event, msg.job_id) for msg in messages if msg.event is not ClientEvent.progress
    ] == [
        (ClientEvent.output, first.id),
        (ClientEvent.finished, first.id),
        (ClientEvent.output, second.id),
        (ClientEvent.finished, second.id),
    ]
    outputs = [msg.result for msg in messages if msg.event is ClientEvent.output]
    assert outputs == [
        TextOutput("2", "Tags", "1girl, solo", "text/plain"),
        TextOutput("2", "Tags", "1girl, solo", "text/plain"),
    ]
    for msg in messages:
        if msg.event is ClientEvent.finished:
            assert msg.images is not None and len(msg.images) == 0


@pytest.mark.parametrize("kind,executed", [(WorkflowKind.tag, False), (WorkflowKind.custom, True)])
@qtapp
async def test_tag_only_websocket_failures(kind: WorkflowKind, executed: bool):
    client = ComfyClient("http://mock")
    work = _tag_work() if kind is WorkflowKind.tag else WorkflowInput(kind)
    job = JobInfo("tag-failure", work, node_count=2)
    client._waiting_job.set(job)

    async def websocket():
        yield _event("execution_start", job.id)
        if kind is WorkflowKind.tag:
            yield _event("execution_cached", job.id, nodes=["1", "2"])
        if executed:
            yield _event("executed", job.id, node="2", output={"tags": ["ignored"]})
        yield _event("executing", job.id, node=None)

    await client._listen_websocket(cast(Any, websocket()))
    messages = _client_messages(client)
    terminal = [msg for msg in messages if msg.event in (ClientEvent.finished, ClientEvent.error)]
    assert len(terminal) == 1
    assert terminal[0].event is ClientEvent.error
