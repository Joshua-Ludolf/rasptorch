import math

import numpy as np
import pytest

import rasptorch
from rasptorch import Tensor
from rasptorch import functional as F
from rasptorch import init as rt_init
from rasptorch import utils
from rasptorch.nn import AvgPool2d, BatchNorm1d, BatchNorm2d, ELU, Embedding, GELU, GRU, LayerNorm, LeakyReLU, MaxPool2d, MultiheadAttention, SiLU
from rasptorch.optim import Adam, AdamW, RMSProp, SGD
from rasptorch.optim_sched import CosineAnnealingLR, ExponentialLR, MultiStepLR, ReduceLROnPlateau, StepLR, WarmupScheduler
from rasptorch import vulkan_backend as vk


def test_new_activation_modules_match_tensor_methods() -> None:
    x0 = np.array([[-2.0, -0.5, 0.5, 2.0]], dtype=np.float32)
    x = Tensor(x0, requires_grad=True)

    np.testing.assert_allclose(GELU()(x).numpy(), x.gelu().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(SiLU()(x).numpy(), x.silu().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(LeakyReLU(0.2)(x).numpy(), x.leaky_relu(0.2).numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ELU(1.5)(x).numpy(), x.elu(1.5).numpy(), rtol=1e-6, atol=1e-6)


def test_binary_cross_entropy_with_logits_backward_matches_formula() -> None:
    x0 = np.array([[0.2, -1.5, 3.0]], dtype=np.float32)
    t0 = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    x = Tensor(x0, requires_grad=True)
    t = Tensor(t0)

    loss = F.binary_cross_entropy_with_logits(x, t)
    loss.backward()

    sig = 1.0 / (1.0 + np.exp(-x0))
    expected = (sig - t0) / x0.size
    np.testing.assert_allclose(x.grad, expected, rtol=1e-6, atol=1e-6)


def test_label_smoothing_zero_matches_cross_entropy() -> None:
    logits_np = np.array([[1.0, 0.0, -1.0], [0.5, -0.25, 0.25]], dtype=np.float32)
    target_np = np.array([0, 2], dtype=np.int64)
    logits = Tensor(logits_np, requires_grad=True)
    target_one_hot = F.one_hot(target_np, 3)

    ce = F.cross_entropy(logits, target_one_hot)
    ls = F.label_smoothing_cross_entropy(logits, target_np, smoothing=0.0)

    np.testing.assert_allclose(ce.numpy(), ls.numpy(), rtol=1e-6, atol=1e-6)


def test_cosine_similarity_matches_numpy() -> None:
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32)
    out = F.cosine_similarity(Tensor(a_np), Tensor(b_np), dim=1).numpy()
    expected = np.sum(a_np * b_np, axis=1) / (
        np.linalg.norm(a_np, axis=1) * np.linalg.norm(b_np, axis=1)
    )
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_initializers_and_backend_like_helpers() -> None:
    t = Tensor(np.empty((6, 4), dtype=np.float32))
    rt_init.xavier_uniform_(t)
    fan_in, fan_out = 4, 6
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    assert np.max(np.abs(t.numpy())) <= bound + 1e-5

    buf = vk.to_gpu(np.ones((2, 3), dtype=np.float32))
    try:
        np.testing.assert_allclose(vk.to_cpu(vk.zeros_like(buf)), np.zeros((2, 3), dtype=np.float32))
        np.testing.assert_allclose(vk.to_cpu(vk.ones_like(buf)), np.ones((2, 3), dtype=np.float32))
    finally:
        vk.free(buf)


def test_adaptive_optimizers_update_parameters() -> None:
    p_adam = Tensor(np.array([1.0, -1.0], dtype=np.float32), requires_grad=True)
    p_adam.grad = np.array([0.5, -0.25], dtype=np.float32)
    opt_adam = Adam([p_adam], lr=0.1)
    opt_adam.step()
    np.testing.assert_allclose(p_adam.numpy(), np.array([0.9, -0.9], dtype=np.float32), atol=1e-5, rtol=1e-5)

    p_adamw = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    p_adamw.grad = np.array([0.5], dtype=np.float32)
    opt_adamw = AdamW([p_adamw], lr=0.1, weight_decay=0.1)
    opt_adamw.step()
    assert float(p_adamw.numpy()[0]) < 0.9

    p_rms = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    p_rms.grad = np.array([0.5], dtype=np.float32)
    opt_rms = RMSProp([p_rms], lr=0.1, rho=0.9)
    opt_rms.step()
    assert float(p_rms.numpy()[0]) < 1.0


def test_schedulers_update_learning_rate_sequences() -> None:
    p = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    opt = SGD([p], lr=1.0)

    step = StepLR(opt, step_size=2, gamma=0.5)
    step.step()
    assert opt.lr == 1.0
    step.step()
    assert opt.lr == 0.5

    opt.lr = 1.0
    multi = MultiStepLR(opt, milestones=[1, 3], gamma=0.1)
    multi.step()
    assert opt.lr == 1.0
    multi.step()
    assert np.isclose(opt.lr, 0.1)

    opt.lr = 1.0
    expo = ExponentialLR(opt, gamma=0.5)
    expo.step()
    assert np.isclose(opt.lr, 0.5)

    opt.lr = 1.0
    cosine = CosineAnnealingLR(opt, T_max=4, eta_min=0.0)
    cosine.step()
    assert np.isclose(opt.lr, 1.0)
    cosine.step()
    assert opt.lr < 1.0

    opt.lr = 1.0
    plateau = ReduceLROnPlateau(opt, patience=1, factor=0.5)
    plateau.step(1.0)
    plateau.step(1.0)
    plateau.step(1.0)
    assert np.isclose(opt.lr, 0.5)

    opt.lr = 1.0
    warm = WarmupScheduler(opt, warmup_steps=2, after_scheduler=StepLR(opt, step_size=1, gamma=0.5))
    warm.step()
    assert np.isclose(opt.lr, 0.5)
    warm.step()
    assert np.isclose(opt.lr, 1.0)
    warm.step()
    assert np.isclose(opt.lr, 1.0)
    warm.step()
    assert np.isclose(opt.lr, 0.5)


def test_gradient_utilities_and_regularization() -> None:
    p = Tensor(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)
    p.grad = np.array([3.0, 4.0], dtype=np.float32)
    total_norm = utils.clip_grad_norm_([p], max_norm=2.5)
    assert np.isclose(total_norm, 5.0)
    np.testing.assert_allclose(p.grad, np.array([1.5, 2.0], dtype=np.float32), rtol=1e-6, atol=1e-6)

    utils.clip_grad_value_([p], clip_value=1.25)
    np.testing.assert_allclose(p.grad, np.array([1.25, 1.25], dtype=np.float32), rtol=1e-6, atol=1e-6)

    class DummyModel:
        def __init__(self, param):
            self.param = param

        def parameters(self):
            return [self.param]

    model = DummyModel(Tensor(np.array([1.0, -2.0], dtype=np.float32), requires_grad=True))
    penalty = utils.l1_regularization(model, 0.5) + utils.l2_regularization(model, 0.25)
    penalty.backward()
    expected = 0.5 * np.sign(model.param.numpy()) + 0.5 * model.param.numpy()
    np.testing.assert_allclose(model.param.grad, expected, rtol=1e-6, atol=1e-6)

    tv_input = Tensor(np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32), requires_grad=True)
    tv = utils.total_variation_loss(tv_input)
    assert float(tv.numpy()[0]) == 6.0


def test_gpu_activation_kernels_match_cpu_paths() -> None:
    x_np = np.array([[-2.0, -0.75, 0.25, 1.5]], dtype=np.float32)
    x_cpu = Tensor(x_np.copy(), requires_grad=True)
    x_gpu = Tensor(x_np.copy(), requires_grad=True).to("gpu")

    y_cpu = x_cpu.gelu() + x_cpu.silu() + x_cpu.leaky_relu(0.2) + x_cpu.elu(1.5)
    y_gpu = x_gpu.gelu() + x_gpu.silu() + x_gpu.leaky_relu(0.2) + x_gpu.elu(1.5)
    y_cpu.sum().backward()
    y_gpu.sum().backward()

    np.testing.assert_allclose(y_gpu.numpy(), y_cpu.numpy(), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(vk.to_cpu(x_gpu.grad_vkbuf), x_cpu.grad, rtol=3e-2, atol=3e-2)


def test_batchnorm_modules_train_and_backward() -> None:
    x1 = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    bn1 = BatchNorm1d(2)
    y1 = bn1(x1)
    y1.sum().backward()
    assert x1.grad is not None
    assert bn1.weight is not None and bn1.weight.grad is not None
    assert bn1.bias is not None and bn1.bias.grad is not None

    x2 = Tensor(np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4), requires_grad=True)
    bn2 = BatchNorm2d(2)
    y2 = bn2(x2)
    assert y2.shape == x2.shape
    bn2.eval()
    y2_eval = bn2(x2)
    assert y2_eval.shape == x2.shape


def test_embedding_backward_accumulates_rows() -> None:
    emb = Embedding(5, 3)
    emb.weight.data[...] = np.arange(15, dtype=np.float32).reshape(5, 3)
    idx = np.array([1, 2, 1], dtype=np.int64)
    out = emb(idx)
    out.sum().backward()

    expected = np.zeros((5, 3), dtype=np.float32)
    expected[1] = 2.0
    expected[2] = 1.0
    np.testing.assert_allclose(emb.weight.grad, expected, rtol=1e-6, atol=1e-6)


def test_multihead_attention_backward_populates_grads() -> None:
    rng = np.random.default_rng(0)
    mha = MultiheadAttention(embed_dim=4, num_heads=2)
    q = Tensor(rng.standard_normal((2, 3, 4), dtype=np.float32), requires_grad=True)
    k = Tensor(rng.standard_normal((2, 3, 4), dtype=np.float32), requires_grad=True)
    v = Tensor(rng.standard_normal((2, 3, 4), dtype=np.float32), requires_grad=True)

    out, weights = mha(q, k, v, need_weights=True)
    assert weights.shape == (2, 3, 3)
    out.sum().backward()

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert v.grad is not None and v.grad.shape == v.shape
    assert mha.q_weight.grad is not None
    assert mha.out_weight.grad is not None


def test_gru_forward_and_autograd_limit() -> None:
    gru = GRU(input_size=3, hidden_size=5, batch_first=True)
    x = Tensor(np.ones((2, 4, 3), dtype=np.float32))
    with rasptorch.no_grad():
        out, hidden = gru(x)
    assert out.shape == (2, 4, 5)
    assert hidden.shape == (1, 2, 5)

    x_req = Tensor(np.ones((2, 4, 3), dtype=np.float32), requires_grad=True)
    out2, hidden2 = gru(x_req)
    loss = out2.sum() + hidden2.sum()
    loss.backward()

    assert x_req.grad is not None and x_req.grad.shape == x_req.shape
    assert gru.weight_ih_l0.grad is not None and gru.weight_ih_l0.grad.shape == gru.weight_ih_l0.shape
    assert gru.weight_hh_l0.grad is not None and gru.weight_hh_l0.grad.shape == gru.weight_hh_l0.shape
    assert gru.bias_ih_l0.grad is not None and gru.bias_ih_l0.grad.shape == gru.bias_ih_l0.shape
    assert gru.bias_hh_l0.grad is not None and gru.bias_hh_l0.grad.shape == gru.bias_hh_l0.shape


def test_amp_scaler_and_cast_helpers() -> None:
    t = Tensor(np.array([1.1, -2.2], dtype=np.float32))
    q = np.asarray(np.array([1.1, -2.2], dtype=np.float16), dtype=np.float32)
    np.testing.assert_allclose(t.half().numpy(), q, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(t.half().float().numpy(), q.astype(np.float32), rtol=0.0, atol=0.0)

    p = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    p.grad = np.array([4.0], dtype=np.float32)
    opt = SGD([p], lr=0.25)
    scaler = rasptorch.GradScaler(init_scale=8.0)
    scaler.unscale_(opt)
    np.testing.assert_allclose(p.grad, np.array([0.5], dtype=np.float32), rtol=1e-6, atol=1e-6)

    with rasptorch.amp.autocast("float16"):
        assert rasptorch.amp.get_compute_dtype() == "float16"
    assert rasptorch.amp.get_compute_dtype() == "float32"


def test_tensor_indexing_reductions_and_extrema() -> None:
    x = Tensor(np.array([[1.0, 3.0, 2.0], [4.0, -1.0, 5.0]], dtype=np.float32), requires_grad=True)
    y = x[1:, 1:]
    np.testing.assert_allclose(y.numpy(), np.array([[-1.0, 5.0]], dtype=np.float32))

    loss = y.sum()
    loss.backward()
    np.testing.assert_allclose(
        x.grad,
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    x2 = Tensor(np.array([[1.0, 3.0, 2.0], [4.0, -1.0, 5.0]], dtype=np.float32), requires_grad=True)
    s = x2.sum(axis=1)
    m = x2.mean(axis=0)
    z = s.sum() + m.sum()
    z.backward()
    expected = np.ones_like(x2.numpy(), dtype=np.float32) + np.full_like(x2.numpy(), 0.5, dtype=np.float32)
    np.testing.assert_allclose(x2.grad, expected, rtol=1e-6, atol=1e-6)

    x3 = Tensor(np.array([[1.0, 3.0, 3.0], [4.0, -1.0, 5.0]], dtype=np.float32), requires_grad=True)
    max_loss = x3.max(axis=1).sum()
    min_loss = x3.min()
    (max_loss + min_loss).backward()
    np.testing.assert_allclose(
        x3.grad,
        np.array([[0.0, 0.5, 0.5], [0.0, 1.0, 1.0]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert x3.argmax() == 5
    np.testing.assert_array_equal(x3.argmin(axis=1), np.array([0, 1], dtype=np.int64))


def test_pooling_layers_forward_and_backward() -> None:
    x_max = Tensor(np.array([[[[1.0, 3.0], [2.0, 4.0]]]], dtype=np.float32), requires_grad=True)
    max_pool = MaxPool2d(2)
    y_max = max_pool(x_max)
    np.testing.assert_allclose(y_max.numpy(), np.array([[[[4.0]]]], dtype=np.float32), rtol=1e-6, atol=1e-6)
    y_max.sum().backward()
    np.testing.assert_allclose(
        x_max.grad,
        np.array([[[[0.0, 0.0], [0.0, 1.0]]]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    x_avg = Tensor(np.array([[[[1.0, 3.0], [2.0, 4.0]]]], dtype=np.float32), requires_grad=True)
    avg_pool = AvgPool2d(2)
    y_avg = avg_pool(x_avg)
    np.testing.assert_allclose(y_avg.numpy(), np.array([[[[2.5]]]], dtype=np.float32), rtol=1e-6, atol=1e-6)
    y_avg.sum().backward()
    np.testing.assert_allclose(
        x_avg.grad,
        np.full((1, 1, 2, 2), 0.25, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_layernorm_gpu_falls_back_for_nondefault_eps() -> None:
    x = Tensor(np.array([[1.0, 2.0, 3.0], [0.5, 1.5, -0.5]], dtype=np.float32), requires_grad=True).to("gpu")
    ln = LayerNorm(3, eps=1e-4).to("gpu")
    y = ln(x)
    y.sum().backward()

    assert y.shape == x.shape
    np.testing.assert_allclose(y.numpy().mean(axis=1), np.zeros((2,), dtype=np.float32), atol=1e-4)
    assert x.grad_vkbuf is not None
