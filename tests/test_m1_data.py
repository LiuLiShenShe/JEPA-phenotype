"""M1 leakage, immutable split, cache and train-only scaling contracts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datasets.m1_feature_dataset import FeatureStore, M1FeatureDataset, M1Index
from training.common import TrainStandardizer, file_hash, object_hash, write_json


ROOT = Path(__file__).resolve().parents[1]


def test_m1_index_history_time_order_and_split(workflow):
    index = M1Index(workflow["root"])
    assert np.all(np.diff(index.indices,axis=1)==1)
    assert np.all(np.diff(index.timestamps,axis=1)>np.timedelta64(0,"s"))
    assert index.rows.groupby("plant_uid").split.nunique().max()==1
    assert set(index.split["train_plants"]).isdisjoint(index.split["val_plants"])
    assert set(index.split["test_plants"]).isdisjoint(index.split["train_plants"]+index.split["val_plants"])


def test_m1_target_path_cannot_enter_history(workflow):
    path = workflow["windows_path"]
    rows = pd.read_csv(path)
    frames = pd.read_csv(workflow["metadata_dir"] / "frames.csv")
    row = rows.iloc[0]
    target = frames.loc[frames.plant_uid.eq(row.plant_uid)&frames.frame_idx.eq(row.target_frame_idx)].iloc[0]
    paths = json.loads(row.input_image_paths)
    paths[-1] = target.rgb_path
    rows.loc[0,"input_image_paths"] = json.dumps(paths)
    rows.to_csv(path,index=False)
    with pytest.raises(ValueError,match="foreign RGB"):
        M1Index(workflow["root"])


def test_standardizer_only_fits_train():
    values = np.array([[1.,2.],[3.,6.]],dtype=np.float32)
    rows = pd.DataFrame({"split":["train","train"],"plant_uid":["p1","p2"],"window_id":["a","b"]})
    scaler = TrainStandardizer.fit(values,rows)
    np.testing.assert_array_equal(scaler.mean,[2,4])
    for split in ("val","test"):
        with pytest.raises(ValueError,match="training"):
            TrainStandardizer.fit(values,rows.assign(split=split))
    np.testing.assert_allclose(scaler.inverse(scaler.transform(values)),values)


def make_cache(directory):
    features = np.arange(4*384,dtype=np.float32).reshape(4,384)/1000
    np.save(directory / "features.npy",features)
    manifest = pd.DataFrame({"feature_row":np.arange(4),"rgb_path":[f"rgb{position}.png" for position in range(4)],
                             "rgb_sha256":["a"*64]*4,"feature_sha256":[hashlib.sha256(row.tobytes()).hexdigest() for row in features]})
    manifest.to_csv(directory / "manifest.csv",index=False)
    metadata = {"status":"complete","feature_dimension":384,"preprocessing":{"version":1},
                "preprocessing_sha256":object_hash({"version":1}),"features_sha256":file_hash(directory / "features.npy"),
                "manifest_sha256":file_hash(directory / "manifest.csv")}
    write_json(directory / "metadata.json",metadata)
    return features,manifest,metadata


def test_cache_maps_original_path_and_rejects_wrong_row(tmp_path):
    features,manifest,metadata = make_cache(tmp_path)
    store = FeatureStore(tmp_path)
    assert store.row_indices([["rgb2.png","rgb0.png","rgb3.png"]]).tolist()==[[2,0,3]]
    np.testing.assert_array_equal(store.features[2],features[2])
    manifest.loc[[0,1],"feature_sha256"] = manifest.loc[[1,0],"feature_sha256"].to_numpy()
    manifest.to_csv(tmp_path / "manifest.csv",index=False)
    metadata["manifest_sha256"] = file_hash(tmp_path / "manifest.csv")
    write_json(tmp_path / "metadata.json",metadata)
    with pytest.raises(ValueError,match="wrong image"):
        FeatureStore(tmp_path)


def test_cache_rejects_preprocessing_change(tmp_path):
    _,_,metadata = make_cache(tmp_path)
    metadata["preprocessing"] = {"version":2}
    write_json(tmp_path / "metadata.json",metadata)
    with pytest.raises(ValueError,match="preprocessing"):
        FeatureStore(tmp_path)


def test_cache_rejects_nan_inf_and_matrix_tampering(tmp_path):
    features,_,metadata = make_cache(tmp_path)
    features[0,0] = np.nan
    np.save(tmp_path / "features.npy",features)
    with pytest.raises(ValueError,match="Non-finite"):
        FeatureStore(tmp_path)
    features[0,0] = 1234
    np.save(tmp_path / "features.npy",features)
    with pytest.raises(ValueError,match="matrix hash"):
        FeatureStore(tmp_path)


def test_feature_inputs_independent_of_targets_and_never_open_images(workflow,monkeypatch):
    index = M1Index(workflow["root"])
    paths = sorted(set(index.paths.ravel()))
    class Store:
        features = np.arange(len(paths)*384,dtype=np.float32).reshape(len(paths),384)
        def row_indices(self, histories):
            lookup = {path:position for position,path in enumerate(paths)}
            return np.asarray([[lookup[path] for path in history] for history in histories])
    from PIL import Image
    monkeypatch.setattr(Image,"open",lambda *args,**kwargs:pytest.fail("Cached model must never read RGB/mask at inference"))
    dataset = M1FeatureDataset(index,Store(),"test",4)
    before = dataset.inputs(0)
    dataset.targets[:] = 999999
    after = dataset.inputs(0)
    assert set(after)=={"features","time_conditions"}
    for key in before:
        assert np.array_equal(before[key].numpy(),after[key].numpy())


def test_full_real_cache_and_frozen_m0_files():
    cache_path = ROOT / "reports/m1/feature_cache.json"
    if not cache_path.exists():
        pytest.skip("Requires local M1 extraction artifacts")
    cache = json.loads(cache_path.read_text())
    store = FeatureStore(ROOT / cache["directory"],root=ROOT)
    assert store.features.shape==(17056,384)
    snapshot = json.loads((ROOT / "reports/m1/m0_snapshot.json").read_text())
    for path,digest in snapshot["files"].items():
        assert file_hash(ROOT / path)==digest,path
