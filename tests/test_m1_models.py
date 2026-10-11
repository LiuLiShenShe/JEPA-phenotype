"""Model shapes, historical-input boundaries, devices and save/reload determinism."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.dinov2_gru import DINOv2GRU
from models.dinov2_mlp import DINOv2MLP
from models.growth_baseline import LinearGrowth, TimeOnly
from models.persistence import Persistence
from training.common import require_device, seed_everything
from training.trainer import load_head, make_model, train_head


@pytest.mark.parametrize("constructor",[DINOv2MLP,DINOv2GRU])
def test_model_shapes_and_checkpoint_reload(constructor,tmp_path):
    seed_everything(42)
    model = constructor().eval()
    features = torch.randn(5,3,384)
    times = torch.randn(5,10)
    predicted = model(features,times)
    assert predicted.shape==(5,2)
    assert next(model.parameters()).device.type=="cpu"
    torch.save(model.state_dict(),tmp_path / "weights.pt")
    loaded = constructor().eval()
    loaded.load_state_dict(torch.load(tmp_path / "weights.pt",weights_only=True))
    torch.testing.assert_close(predicted,loaded(features,times),rtol=0,atol=0)
    torch.testing.assert_close(loaded(features,times),loaded(features,times),rtol=0,atol=0)


def test_single_frame_mlp_cannot_use_earlier_rgb():
    model = DINOv2MLP().eval()
    features = torch.randn(8,3,384)
    changed = features.clone()
    changed[:,:2] += 1000
    times = torch.randn(8,10)
    torch.testing.assert_close(model(features,times),model(changed,times),rtol=0,atol=0)


def test_gru_uses_chronological_history():
    model = DINOv2GRU().eval()
    features = torch.randn(8,3,384)
    times = torch.randn(8,10)
    assert not torch.allclose(model(features,times),model(features.flip(1),times))


@pytest.mark.parametrize("device_name",["cuda:0","cuda:1"])
@pytest.mark.parametrize("constructor",[DINOv2MLP,DINOv2GRU])
def test_real_gpu_device_and_gradients(device_name,constructor):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    seed_everything(42)
    model = constructor().to(device_name)
    features = torch.randn(4,3,384,device=device_name)
    times = torch.randn(4,10,device=device_name)
    output = model(features,times)
    assert str(output.device)==device_name
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_oracle_baselines_use_real_elapsed_time():
    times = np.array([[0,1,3],[0,2,3]],dtype=float)
    slopes = np.array([2.,.1])
    histories = 1+times[:,:,None]*slopes
    histories[:,:,1] /= 10
    predicted = LinearGrowth().predict(histories,times,np.array([2,2]))
    np.testing.assert_allclose(predicted,[[11,.15],[11,.15]])
    np.testing.assert_allclose(Persistence().predict(histories),histories[:,-1])


def test_time_only_checkpoint_exact_reload():
    random = np.random.default_rng(42)
    times = random.normal(size=(64,10)).astype(np.float32)
    targets = np.column_stack([random.uniform(5,20,64),random.uniform(.1,.5,64)]).astype(np.float32)
    rows = pd.DataFrame({"split":"train","plant_uid":[f"p{position//8}" for position in range(64)],"window_id":[f"w{position}" for position in range(64)]})
    model = TimeOnly(10).fit(times,targets,rows)
    loaded = TimeOnly.from_state(json.loads(json.dumps(model.state())))
    np.testing.assert_array_equal(model.predict(times),loaded.predict(times))


def test_trainer_refuses_test_for_model_selection(tmp_path):
    class InvalidDataset:
        rows = pd.DataFrame({"split":["test"]})
    with pytest.raises(ValueError,match="never test"):
        train_head(InvalidDataset(),InvalidDataset(),{},tmp_path)


def test_full_real_checkpoints_and_repeatability():
    root = Path(__file__).resolve().parents[1]
    path = root / "reports/m1/training_complete.json"
    if not path.exists():
        pytest.skip("Requires completed real main experiments")
    selection = json.loads((root / "reports/m1/model_selection.json").read_text())
    assert selection["test_used_for_selection"] is False
    assert all(candidate["selection_split"]=="val" for candidate in selection["candidates"])
    count = 0
    for path in sorted((root / "outputs/m1/checkpoints").glob("B2*/checkpoint.pt")):
        model,checkpoint = load_head(path,"cpu")
        assert checkpoint["target_scaler"]["provenance"]["split"]=="train"
        assert checkpoint["time_scaler"]["provenance"]["split"]=="train"
        assert checkpoint["test_used_for_selection"] is False
        inputs = torch.ones(3,3,384)
        times = torch.ones(3,10)
        torch.testing.assert_close(model(inputs,times),model(inputs,times),rtol=0,atol=0)
        count += 1
    assert count==18


def test_actual_checkpoint_scalers_match_only_the_training_windows():
    root = Path(__file__).resolve().parents[1]
    if not (root / "reports/m1/training_complete.json").exists():
        pytest.skip("Requires completed real experiments")
    from datasets.m1_feature_dataset import M1Index
    from training.common import object_hash
    index = M1Index(root)
    for horizon in (2,4,8):
        positions = index.select("train",horizon)
        rows = index.rows.iloc[positions]
        expected_window_hash = object_hash(sorted(rows.window_id.tolist()))
        for model_name in ("B2a","B2b"):
            path = root / f"outputs/m1/checkpoints/{model_name}_h{horizon}_s42/checkpoint.pt"
            _,checkpoint = load_head(path,"cpu")
            for key,values in (("target_scaler",index.targets[positions]),("time_scaler",index.times[positions])):
                state = checkpoint[key]
                assert state["provenance"]["window_ids_sha256"]==expected_window_hash
                assert state["provenance"]["plants"]==219
                np.testing.assert_array_equal(np.array(state["mean"],dtype=np.float32),values.mean(axis=0))
                np.testing.assert_array_equal(np.array(state["scale"],dtype=np.float32),np.maximum(values.std(axis=0),1e-6))


def test_actual_test_predictions_and_selection_registry_are_complete():
    root = Path(__file__).resolve().parents[1]
    if not (root / "reports/m1/evaluation_complete.json").exists():
        pytest.skip("Requires completed real evaluation")
    from training.common import file_hash
    selection_path = root / "reports/m1/model_selection.json"
    selection = json.loads(selection_path.read_text())
    split = json.loads((root / "splits/phenoleaf_plant_split_seed42.json").read_text())
    selection_hash = file_hash(selection_path)
    train_plants = set(split["train_plants"])
    val_plants = set(split["val_plants"])
    test_plants = set(split["test_plants"])
    repeatability = json.loads((root / "reports/m1/inference_repeatability.json").read_text())
    assert repeatability["selection_registry_sha256"]==selection_hash
    assert len(repeatability["runs"])==18
    assert all(run["max_difference"]==0 for run in repeatability["runs"])
    for model_name in ("B0","B1","B1b","B2a","B2b"):
        for horizon in (2,4,8):
            seeds = (42,43,44) if model_name in {"B2a","B2b"} else (42,)
            for seed in seeds:
                run_name = f"{model_name}_h{horizon}" + (f"_s{seed}" if model_name in {"B2a","B2b"} else "")
                rows = pd.read_csv(root / f"outputs/m1/predictions/{run_name}/test.csv")
                assert set(rows.plant_uid)==test_plants
                assert set(rows.split)=={"test"}
                assert rows.window_id.is_unique
                assert len(rows)=={2:2075,4:1983,8:1799}[horizon]
                if model_name in {"B2a","B2b"}:
                    _,checkpoint = load_head(root / f"outputs/m1/checkpoints/{run_name}/checkpoint.pt","cpu")
                    assert checkpoint["config"]["selection_registry_sha256"]==selection_hash
                    assert set(checkpoint["train_plants"])==train_plants
                    assert set(checkpoint["val_plants"])==val_plants
                    assert test_plants.isdisjoint(checkpoint["train_plants"]+checkpoint["val_plants"])
    assert selection["split_sha256"]==file_hash(root / "splits/phenoleaf_plant_split_seed42.json")


def test_refuse_silent_cuda_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="silent CPU fallback"):
        require_device("cuda:0")


def test_official_encoder_real_cache_and_letterboxing():
    root = Path(__file__).resolve().parents[1]
    if not (root / "reports/m1/feature_cache.json").exists():
        pytest.skip("Requires official pretrained cache artifacts")
    from extract_dinov2_features import load_encoder,preprocess
    from PIL import Image
    from datasets.m1_feature_dataset import FeatureStore
    cache = json.loads((root / "reports/m1/feature_cache.json").read_text())
    store = FeatureStore(root / cache["directory"])
    tensor = preprocess(Image.new("RGB",(400,200),(255,255,255)))
    assert tensor.shape==(3,224,224)
    assert torch.all(tensor[:,80,80]>1)
    assert torch.all(tensor[:,0,80].abs()<.02)
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable for real encoder")
    encoder = load_encoder(root,torch.device("cuda:0"))
    assert not encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    positions = [0,8000,17055]
    tensors = []
    for position in positions:
        with Image.open(root / store.manifest.iloc[position].rgb_path) as image:
            tensors.append(preprocess(image))
    with torch.inference_mode():
        actual = encoder(torch.stack(tensors).to("cuda:0")).cpu().numpy()
    assert actual.shape==(3,384)
    np.testing.assert_allclose(actual,store.features[positions],rtol=2e-5,atol=2e-5)
