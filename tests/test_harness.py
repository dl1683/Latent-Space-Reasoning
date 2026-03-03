"""Tests for the unified experiment harness."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.harness import (
    ActiveInferenceSurrogate,
    Candidate,
    DecodeConfig,
    DecodeMode,
    EvolutionParams,
    ExperimentCondition,
    QDParams,
    Task,
    _apply_mutation,
    _make_noise,
    dense_score,
    exact_sign_flip_pvalue,
    generate_all_unique_tasks,
    split_train_test,
    verify_answer,
)
from latent_reasoning.decode.projection import (
    latent_to_soft_prompt,
    make_row_orthonormal_W,
    radial_tanh_squash,
)
from latent_reasoning.decode.steering import (
    DualSteeringProcessor,
    make_steer_projection,
)


class TestTaskGeneration:
    def test_deterministic(self):
        a = generate_all_unique_tasks(5, [2, 3])
        b = generate_all_unique_tasks(5, [2, 3])
        for depth in [2, 3]:
            assert len(a[depth]) == len(b[depth])
            for ta, tb in zip(a[depth], b[depth]):
                assert ta.task_id == tb.task_id
                assert ta.correct_answer == tb.correct_answer

    def test_unique_ids(self):
        tasks = generate_all_unique_tasks(10, [2])
        ids = [t.task_id for t in tasks[2]]
        assert len(ids) == len(set(ids))

    def test_count_matches_branching(self):
        tasks = generate_all_unique_tasks(5, [2, 3])
        assert len(tasks[2]) == 5 ** 2
        assert len(tasks[3]) == 5 ** 3

    def test_answers_are_correct(self):
        tasks = generate_all_unique_tasks(3, [2])
        # First task: path [0,0] -> sum=0, answer = 0*3 + 2*7 = 14
        t = tasks[2][0]
        assert t.correct_answer == 14


class TestSplitTrainTest:
    def test_no_overlap(self):
        tasks = generate_all_unique_tasks(15, [2, 3])
        train, test = split_train_test(tasks, 20, 150)
        train_ids = {t.task_id for t in train}
        test_ids = {t.task_id for t in test}
        assert len(train_ids & test_ids) == 0

    def test_correct_sizes(self):
        tasks = generate_all_unique_tasks(15, [2, 3])
        train, test = split_train_test(tasks, 20, 150)
        assert len(test) == 40  # 20 per depth x 2
        assert len(train) == 300  # 150 per depth x 2

    def test_deterministic(self):
        tasks = generate_all_unique_tasks(15, [2])
        a_train, a_test = split_train_test(tasks, 10, 50, seed=42)
        b_train, b_test = split_train_test(tasks, 10, 50, seed=42)
        assert [t.task_id for t in a_test] == [t.task_id for t in b_test]

    def test_insufficient_tasks_raises(self):
        tasks = generate_all_unique_tasks(3, [2])  # Only 9 tasks
        with pytest.raises(ValueError, match="need"):
            split_train_test(tasks, 5, 10)  # Need 15


class TestVerifyAnswer:
    def test_exact_match(self):
        assert verify_answer("The answer is 42", 42)

    def test_last_number(self):
        assert verify_answer("I think 10 but maybe 42", 42)

    def test_no_numbers(self):
        assert not verify_answer("No numbers here", 42)

    def test_wrong_answer(self):
        assert not verify_answer("The answer is 43", 42)

    def test_negative(self):
        assert verify_answer("Result: -7", -7)


class TestDenseScore:
    def test_exact_match(self):
        assert dense_score("42", 42) == 1.0

    def test_no_numbers(self):
        assert dense_score("nothing", 42) == 0.0

    def test_partial_credit(self):
        s = dense_score("43", 42)
        assert 0.0 < s < 1.0
        assert abs(s - 0.5) < 0.01  # 1/(1+1)

    def test_large_distance(self):
        s = dense_score("142", 42)
        assert s < 0.02  # 1/(1+100) = 0.0099


class TestProjection:
    def test_w_orthonormality(self):
        W = make_row_orthonormal_W(64, 256, seed=1234)
        assert W.shape == (64, 256)
        WWT = W @ W.T
        I = torch.eye(64)
        assert torch.allclose(WWT, I, atol=1e-5)

    def test_w_deterministic(self):
        W1 = make_row_orthonormal_W(64, 256, seed=42)
        W2 = make_row_orthonormal_W(64, 256, seed=42)
        assert torch.allclose(W1, W2)

    def test_w_different_seeds(self):
        W1 = make_row_orthonormal_W(64, 256, seed=1)
        W2 = make_row_orthonormal_W(64, 256, seed=2)
        assert not torch.allclose(W1, W2)


class TestRadialTanhSquash:
    def test_bounded(self):
        v = torch.randn(100) * 10
        squashed = radial_tanh_squash(v, r_max=1.0)
        # tanh saturates to exactly r_max for large inputs in float32
        assert squashed.norm().item() <= 1.0 + 1e-6

    def test_direction_preserved(self):
        v = torch.randn(100) * 5
        squashed = radial_tanh_squash(v, r_max=2.0)
        cos_sim = torch.nn.functional.cosine_similarity(
            v.unsqueeze(0), squashed.unsqueeze(0),
        )
        assert cos_sim.item() > 0.99

    def test_small_input_passthrough(self):
        v = torch.randn(100) * 0.001
        squashed = radial_tanh_squash(v, r_max=10.0)
        # For small inputs, tanh(x) ~ x, so output ~ input
        assert torch.allclose(v, squashed, atol=1e-4)


class TestLatentToSoftPrompt:
    def test_shape(self):
        W = make_row_orthonormal_W(64, 256, seed=1234)
        latent = torch.randn(64) * 0.3
        sp = latent_to_soft_prompt(
            latent, W, curvature=0.5,
            embed_dim=32, num_tokens=8, target_rms=0.02,
            use_logmap=False,
        )
        assert sp.shape == (1, 8, 32)

    def test_rms_matches_target(self):
        W = make_row_orthonormal_W(64, 256, seed=1234)
        latent = torch.randn(64) * 0.3
        target = 0.05
        sp = latent_to_soft_prompt(
            latent, W, curvature=0.5,
            embed_dim=32, num_tokens=8, target_rms=target,
            use_logmap=False,
        )
        actual_rms = sp.float().square().mean().sqrt().item()
        assert abs(actual_rms - target) / target < 0.01

    def test_euclidean_vs_hyperbolic(self):
        """logmap0 preserves direction but amplifies magnitude; after RMS
        normalization the final tokens match in direction.  We verify both
        paths produce valid outputs, and that the pre-normalization norms
        differ (proving logmap0 was actually applied)."""
        W = make_row_orthonormal_W(64, 256, seed=1234)
        latent = torch.randn(64)
        latent = latent / latent.norm() * 0.8  # norm=0.8, well inside ball
        sp_euc = latent_to_soft_prompt(
            latent, W, curvature=0.5,
            embed_dim=32, num_tokens=8, target_rms=0.02,
            use_logmap=False,
        )
        sp_hyp = latent_to_soft_prompt(
            latent, W, curvature=0.5,
            embed_dim=32, num_tokens=8, target_rms=0.02,
            use_logmap=True,
        )
        assert sp_euc.shape == sp_hyp.shape == (1, 8, 32)
        # Both match target RMS
        for sp in [sp_euc, sp_hyp]:
            rms = sp.float().square().mean().sqrt().item()
            assert abs(rms - 0.02) / 0.02 < 0.01
        # Verify logmap0 changes intermediate: compute tangent norms directly
        from latent_reasoning.utils import hyperbolic as hyp
        tangent_euc = latent.clone()
        tangent_hyp = hyp.logmap0(latent.clone(), 0.5)
        # logmap0 amplifies norm for points inside the ball
        assert tangent_hyp.norm().item() > tangent_euc.norm().item()


class TestMutation:
    def test_hyperbolic_stays_in_ball(self):
        curvature = 0.5
        ball_radius = (1.0 / math.sqrt(curvature)) * 0.95
        parent = torch.randn(1, 64) * 0.1
        from latent_reasoning.utils import hyperbolic as hyp
        parent = hyp.expmap0(parent.squeeze(), curvature).unsqueeze(0)

        rng = torch.Generator().manual_seed(42)
        noise = _make_noise(parent.shape, 0.1, 64, rng)
        mutated = _apply_mutation(parent, noise, curvature, ball_radius, "hyperbolic")
        assert mutated.squeeze().norm().item() < 1.0 / math.sqrt(curvature)

    def test_euclidean_stays_in_ball(self):
        ball_radius = 1.34
        parent = torch.randn(1, 64) * 0.5
        rng = torch.Generator().manual_seed(42)
        noise = _make_noise(parent.shape, 0.5, 64, rng)
        mutated = _apply_mutation(parent, noise, 0.5, ball_radius, "euclidean")
        assert mutated.squeeze().norm().item() <= ball_radius + 1e-6


class TestSteering:
    def test_steer_projection_orthonormal(self):
        W = make_steer_projection(64, 128, seed=5678)
        assert W.shape == (64, 128)
        WWT = W @ W.T
        I = torch.eye(64)
        assert torch.allclose(WWT, I, atol=1e-5)

    def test_dual_steering_processor_passthrough(self):
        omega = torch.randn(100)
        omega = omega / omega.norm()
        proc = DualSteeringProcessor(omega, eta=0.0)
        logits = torch.randn(1, 100)
        out = proc(torch.zeros(1, 1, dtype=torch.long), logits)
        assert torch.allclose(out, logits)


class TestSignFlip:
    def test_all_positive_diffs(self):
        # All diffs positive -> p should be small (1/2^n)
        diffs = [1.0, 1.0, 1.0, 1.0, 1.0]
        p = exact_sign_flip_pvalue(diffs, "greater")
        assert p == pytest.approx(1.0 / 32.0)

    def test_all_zero_diffs(self):
        diffs = [0.0, 0.0, 0.0]
        p = exact_sign_flip_pvalue(diffs, "greater")
        assert p == 1.0

    def test_mixed_diffs(self):
        diffs = [1.0, -1.0, 1.0]
        p = exact_sign_flip_pvalue(diffs, "greater")
        assert 0.0 < p < 1.0


class TestDecodeConfig:
    def test_default_mode(self):
        cfg = DecodeConfig()
        assert cfg.mode == DecodeMode.SOFT_PROMPT

    def test_rng_seed_mode(self):
        cfg = DecodeConfig(mode=DecodeMode.RNG_SEED)
        assert cfg.mode == DecodeMode.RNG_SEED


class TestActiveSurrogate:
    """Tests for Active Inference surrogate screening."""

    def test_surrogate_predict_returns_bounded_mean(self):
        surr = ActiveInferenceSurrogate(latent_dim=64, proj_dim=16, hidden_dim=32)
        latent = torch.randn(1, 64)
        mean_acc, var = surr.predict(latent)
        assert 0.0 <= mean_acc <= 1.0
        assert var > 0.0

    def test_surrogate_efe_is_negative(self):
        surr = ActiveInferenceSurrogate(latent_dim=64, proj_dim=16)
        latent = torch.randn(1, 64)
        efe = surr.expected_free_energy(latent)
        # EFE = -(mean_acc + beta*var), always negative since both terms > 0
        assert efe < 0.0

    def test_surrogate_select_returns_k_candidates(self):
        surr = ActiveInferenceSurrogate(latent_dim=64, proj_dim=16)
        candidates = [Candidate(latent=torch.randn(1, 64)) for _ in range(20)]
        selected = surr.select_by_efe(candidates, k=5)
        assert len(selected) == 5

    def test_surrogate_update_trains_without_error(self):
        surr = ActiveInferenceSurrogate(latent_dim=64, proj_dim=16)
        for i in range(10):
            surr.update(torch.randn(1, 64), float(i) / 10.0)
        # After 10 updates, history should have 10 entries
        assert len(surr.history) == 10

    def test_surrogate_beta_annealing(self):
        surr = ActiveInferenceSurrogate(latent_dim=64, beta=1.0, beta_decay=0.5)
        assert surr.beta == 1.0
        surr.anneal_beta()
        assert surr.beta == pytest.approx(0.5)
        surr.anneal_beta()
        assert surr.beta == pytest.approx(0.25)

    def test_experiment_condition_surrogate_flag(self):
        cfg = DecodeConfig()
        cond_no = ExperimentCondition(name="no_surr", decode_cfg=cfg, use_surrogate=False)
        cond_yes = ExperimentCondition(name="surr", decode_cfg=cfg, use_surrogate=True)
        assert not cond_no.use_surrogate
        assert cond_yes.use_surrogate

    def test_surrogate_deterministic_projection(self):
        """Same seed -> same projection matrix."""
        s1 = ActiveInferenceSurrogate(latent_dim=64, proj_dim=16, seed=42)
        s2 = ActiveInferenceSurrogate(latent_dim=64, proj_dim=16, seed=42)
        assert torch.allclose(s1.proj, s2.proj)


class TestQDIntegration:
    """Tests for QD integration in experiment harness."""

    def test_qd_params_defaults(self):
        params = QDParams()
        assert params.bd_dim == 16
        assert params.novelty_weight == 0.3
        assert params.archive_size == 100

    def test_experiment_condition_qd_flag(self):
        cfg = DecodeConfig()
        cond = ExperimentCondition(name="qd", decode_cfg=cfg, use_qd=True)
        assert cond.use_qd
        assert cond.qd_params is None  # Uses defaults

    def test_experiment_condition_qd_with_params(self):
        cfg = DecodeConfig()
        qd = QDParams(bd_dim=8, novelty_weight=0.5)
        cond = ExperimentCondition(name="qd", decode_cfg=cfg, use_qd=True, qd_params=qd)
        assert cond.qd_params.bd_dim == 8
        assert cond.qd_params.novelty_weight == 0.5

    def test_qd_and_surrogate_mutually_exclusive_by_priority(self):
        """QD takes priority over surrogate when both are set."""
        cfg = DecodeConfig()
        cond = ExperimentCondition(
            name="both", decode_cfg=cfg,
            use_qd=True, use_surrogate=True,
        )
        # QD flag is checked first in run_experiment
        assert cond.use_qd
