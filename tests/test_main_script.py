import sys

import main as main_script


class _Backend:
    def __init__(self, name: str) -> None:
        self.name = name


def test_main_gpu_passes_seed_to_vulkan_trainer(monkeypatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(main_script, "connect_backend", lambda *args, **kwargs: _Backend("vulkan"))

    def _fake_train(x, y, **kwargs):
        seen["seed"] = kwargs.get("seed")

    monkeypatch.setattr(main_script, "train_mlp_regression_gpu", _fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--device", "gpu", "--epochs", "1", "--seed", "123"],
    )

    main_script.main()
    assert seen["seed"] == 123


def test_main_creates_rng_from_seed(monkeypatch) -> None:
    """Test that main.py creates a np.random.Generator with the correct seed."""
    rng_configs: list[int] = []

    original_default_rng = main_script.np.random.default_rng

    def _fake_default_rng(seed=None):
        if seed is not None:
            rng_configs.append(seed)
        return original_default_rng(seed)

    monkeypatch.setattr(main_script.np.random, "default_rng", _fake_default_rng)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--device", "cpu", "--epochs", "0", "--seed", "77"],
    )

    main_script.main()
    # Verify that default_rng was called with seed=77
    assert 77 in rng_configs
