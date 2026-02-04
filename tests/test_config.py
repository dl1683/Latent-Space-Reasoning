"""Tests for configuration system."""

import pytest
import tempfile
from pathlib import Path

from latent_reasoning.config import (
    Config,
    EncoderConfig,
    JudgeConfig,
    ScorerConfig,
    ModifierConfig,
    EvolutionConfig,
    BudgetConfig,
    get_default_config,
)


class TestEncoderConfig:
    def test_default_values(self):
        config = EncoderConfig()
        assert config.model == "Qwen/Qwen3-1.7B"
        assert config.layer == -4
        assert config.pooling == "mean"
        assert config.device == "auto"
        assert config.latent_dim == 1024

    def test_custom_values(self):
        config = EncoderConfig(
            model="bert-base-uncased",
            layer=-2,
            pooling="cls",
            device="cuda",
        )
        assert config.model == "bert-base-uncased"
        assert config.layer == -2
        assert config.pooling == "cls"
        assert config.device == "cuda"


class TestJudgeConfig:
    def test_empty_judges(self):
        config = JudgeConfig(scorers=[], modifiers=[])
        assert config.scorers == []
        assert config.modifiers == []
        assert config.aggregation == "mean"

    def test_with_scorers(self):
        config = JudgeConfig(
            scorers=[
                ScorerConfig(model="bert-base-uncased"),
                ScorerConfig(model="roberta-base"),
            ],
        )
        assert len(config.scorers) == 2
        assert config.scorers[0].model == "bert-base-uncased"


class TestEvolutionConfig:
    def test_default_values(self):
        config = EvolutionConfig()
        assert config.chains == 10
        assert config.generations == 30
        assert config.temperature == 0.5
        assert config.temperature_decay == 0.95

    def test_selection_config(self):
        config = EvolutionConfig()
        assert config.selection.strategy == "elitist"
        assert config.selection.survivors == 5
        assert config.selection.elite == 2

    def test_validation(self):
        # Temperature must be 0-2
        with pytest.raises(Exception):
            EvolutionConfig(temperature=-0.1)


class TestConfig:
    def test_default_config(self):
        config = get_default_config()
        assert config.encoder.model == "Qwen/Qwen3-1.7B"
        assert len(config.judges.scorers) == 1
        assert config.evolution.chains == 5

    def test_from_dict(self):
        data = {
            "encoder": {"model": "custom-model"},
            "evolution": {"chains": 20},
        }
        config = Config.from_dict(data)
        assert config.encoder.model == "custom-model"
        assert config.evolution.chains == 20

    def test_to_dict(self):
        config = get_default_config()
        data = config.to_dict()
        assert "encoder" in data
        assert "evolution" in data
        assert data["encoder"]["model"] == "Qwen/Qwen3-1.7B"

    def test_yaml_roundtrip(self):
        config = get_default_config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            loaded = Config.from_yaml(f.name)

        assert loaded.encoder.model == config.encoder.model
        assert loaded.evolution.chains == config.evolution.chains


class TestBudgetConfig:
    def test_default_values(self):
        config = BudgetConfig()
        assert config.max_evaluations == 500
        assert config.max_time is None

    def test_custom_values(self):
        config = BudgetConfig(max_evaluations=1000, max_time=60.0)
        assert config.max_evaluations == 1000
        assert config.max_time == 60.0
