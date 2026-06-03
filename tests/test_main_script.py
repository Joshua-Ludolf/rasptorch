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


def test_main_sets_numpy_seed(monkeypatch) -> None:
    seen: list[int] = []
    monkeypatch.setattr(main_script.np.random, "seed", lambda value: seen.append(int(value)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--device", "cpu", "--epochs", "0", "--seed", "77"],
    )

    main_script.main()
    assert seen == [77]
