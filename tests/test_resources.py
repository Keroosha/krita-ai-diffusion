import json
from itertools import chain

import ai_diffusion.backend.resources as res
from ai_diffusion.backend.resolution import CheckpointResolution
from ai_diffusion.backend.resources import (
    Arch,
    ControlMode,
    ModelRequirements,
    ModelResource,
    ResourceId,
    ResourceKind,
)
from ai_diffusion.image import Extent

from .config import result_dir


def test_resources_json():
    result = {}
    result["required"] = ModelResource.as_list(res.required_models)
    result["checkpoints"] = ModelResource.as_list(res.default_checkpoints)
    result["upscale"] = ModelResource.as_list(res.upscale_models)
    result["optional"] = ModelResource.as_list(res.optional_models)
    result["prefetch"] = ModelResource.as_list(res.prefetch_models)
    result["deprecated"] = ModelResource.as_list(res.deprecated_models)

    original = res._models_file.read_text()
    result_string = json.dumps(result, indent=2)
    (result_dir / "resources.json").write_text(result_string)
    assert result_string == original


def test_json_no_duplicates():
    d = json.load(res._models_file.open())
    all_ids = set()
    all_names = set()
    for cat in d.values():
        for m in cat:
            assert m["name"] not in all_names, f"Duplicate model name: {m['name']}"
            all_names.add(m["name"])
            ids = m["id"]
            ids = [ids] if isinstance(ids, str) else ids
            for id in ids:
                assert id not in all_ids, f"Duplicate model id: {id}"
            all_ids.update(ids)


def test_same_name_same_model():
    for m in res.all_models(include_deprecated=True):
        for o in res.all_models(include_deprecated=True):
            if m is not o and m.name == o.name:
                assert all(mf.url == of.url for mf, of in zip(m.files, o.files))


def test_resource_ids_exist():
    ids = chain(res.required_resource_ids, res.recommended_resource_ids)
    for resource_id in ids:
        if resource_id.arch in (
            Arch.sd3,
            Arch.qwen,
            Arch.qwen_e,
            Arch.qwen_e_p,
            Arch.flux2_9b,
            Arch.ernie,
            Arch.krea2,
        ):
            continue  # no model downloads yet
        model = res.find_resource(resource_id)
        assert model is not None, f"Resource ID {resource_id} not found"


def test_anima_architecture_and_resolution():
    assert Arch.from_string("anima") is Arch.anima
    assert Arch.from_string("unknown", filename="models/anima-base-v1.0.safetensors") is Arch.anima

    resolution = CheckpointResolution.compute(Extent(1024, 1024), Arch.anima)
    assert resolution == CheckpointResolution(512, 1536, 0.5, 1.5)


def test_anima_managed_resources():
    expected = {
        ResourceId(ResourceKind.checkpoint, Arch.anima, "base"): (
            "anima-base-v1.0.safetensors",
            "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e",
        ),
        ResourceId(ResourceKind.text_encoder, Arch.anima, "qwen_3_06b"): (
            "qwen_3_06b_base.safetensors",
            "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
        ),
        ResourceId(ResourceKind.vae, Arch.anima, "default"): (
            "qwen_image_vae.safetensors",
            "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
        ),
        ResourceId(ResourceKind.model_patch, Arch.anima, ControlMode.universal): (
            "anima-lllite-any-test-like-v2.safetensors",
            "8863424f04a2445815a827b22cbda557a085d2752e05fab54a2b303821a9e3b2",
        ),
        ResourceId(ResourceKind.model_patch, Arch.anima, ControlMode.inpaint): (
            "anima-lllite-inpainting-v2.safetensors",
            "5242e677d2be34ee70ca7c97c3b14ff5ee49838c03fc1e60ac4852a180db6ef5",
        ),
    }
    for id, (filename, sha256) in expected.items():
        model = res.get_resource(id)
        assert model.filename == filename
        assert model.files[0].sha256 == sha256

    adapter = res.get_resource(
        ResourceId(ResourceKind.ip_adapter, Arch.anima, ControlMode.reference)
    )
    assert adapter.requirements is ModelRequirements.cuda
    assert [str(file.path) for file in adapter.files] == [
        "models/ipadapter/ip_adapter-Character_Reference-10.safetensors",
        "models/siglip2/siglip2-base-patch16-512/config.json",
        "models/siglip2/siglip2-base-patch16-512/model.safetensors",
    ]
    assert [file.sha256 for file in adapter.files] == [
        "dc9612cc28b55f00ba39f6169bfe4871130be028a55682ab54dd51360acb8232",
        "a15bc39fbcd92498dea6d5d3cb2b0d8b9ba389d6a3f2b565ea66128a2d141934",
        "fe0e601c625e69eed8e73500d39e9b6164403fe03db8048e87913c3cefbbb3fe",
    ]


def test_anima_search_paths_and_custom_node():
    modes = [
        ControlMode.inpaint,
        ControlMode.universal,
        ControlMode.scribble,
        ControlMode.line_art,
        ControlMode.depth,
        ControlMode.pose,
    ]
    assert all(res.search_path(ResourceKind.model_patch, Arch.anima, mode) for mode in modes)
    assert all(res.search_path(ResourceKind.controlnet, Arch.anima, mode) is None for mode in modes)
    universal = res.search_path(ResourceKind.model_patch, Arch.anima, ControlMode.universal)
    inpaint = res.search_path(ResourceKind.model_patch, Arch.anima, ControlMode.inpaint)
    assert universal is not None and universal[0] == "anima-lllite-any-test-like-v2"
    assert inpaint is not None and inpaint[0] == "anima-lllite-inpainting-v2"
    assert res.search_path(ResourceKind.ip_adapter, Arch.anima, ControlMode.reference) == [
        "ip_adapter-character_reference-10"
    ]

    assert all(node.name != "Anima IP-Adapter" for node in res.required_custom_nodes)
    node = next(node for node in res.optional_custom_nodes if node.name == "Anima IP-Adapter")
    assert node.version == "3813b8c8a655e1a1860b45d9a84ed43383528074"
    assert node.nodes == ["AnimaIPAdapterLoader", "AnimaIPAdapterApply"]

    assert all(node.name != "Anima Regional Conditioning" for node in res.required_custom_nodes)
    node = next(
        node for node in res.optional_custom_nodes if node.name == "Anima Regional Conditioning"
    )
    assert node.folder == "Comfyui-Anima-Regional-Conditioning"
    assert node.url == "https://github.com/Sen-sou/Comfyui-Anima-Regional-Conditioning"
    assert node.version == "099cf1fa052721394963418455d49f7087efaf6c"
    assert node.nodes == ["AnimaConditioningRegion", "ApplyAnimaRegionalConditioningPatch"]
    assert res.version == "1.55.0"
    assert res.comfy_version == "4800e78518ebb1f2a9443ea5418edbff6c3935f9"
