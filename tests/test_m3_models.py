import pytest
import torch

from models.pointnet_forecaster import PointNetEncoder, PointNetForecaster
from models.pointnet_gru_forecaster import PointNetGRUForecaster


@pytest.mark.parametrize("model_class", [PointNetForecaster, PointNetGRUForecaster])
def test_shapes_gradients_and_parameter_budget(model_class):
    torch.set_num_threads(2)
    model = model_class()
    points = torch.randn(4, 3, 64, 3)
    times = torch.randn(4, 7)
    prediction = model(points, times)
    assert prediction.shape == (4, 3)
    prediction.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) < 250000
    with pytest.raises(ValueError):
        model(points[:, :2], times)


def test_pointnet_permutation_invariance():
    encoder = PointNetEncoder().eval()
    points = torch.randn(2, 80, 3)
    torch.testing.assert_close(encoder(points), encoder(points[:, torch.randperm(80)]), atol=1e-6, rtol=1e-5)


def test_repeated_latest_same_capacity_and_exact_evaluation_control():
    real = PointNetGRUForecaster().eval()
    repeated = PointNetGRUForecaster(repeated_latest=True).eval()
    repeated.load_state_dict(real.state_dict())
    assert list(real.state_dict()) == list(repeated.state_dict())
    points = torch.randn(3, 3, 40, 3)
    times = torch.randn(3, 7)
    reference = real(points[:, -1:].repeat(1, 3, 1, 1), times)
    torch.testing.assert_close(reference, repeated(points, times), atol=1e-6, rtol=1e-5)
    changed = points.clone()
    changed[:, :2] += 500
    torch.testing.assert_close(repeated(changed, times), repeated(points, times))
    single = PointNetForecaster().eval()
    torch.testing.assert_close(single(changed, times), single(points, times))


def test_repeated_latest_is_independently_trainable():
    model = PointNetGRUForecaster(repeated_latest=True)
    prediction = model(torch.randn(4, 3, 24, 3), torch.randn(4, 7))
    prediction.square().mean().backward()
    assert model.gru.weight_ih_l0.grad.abs().sum() > 0
