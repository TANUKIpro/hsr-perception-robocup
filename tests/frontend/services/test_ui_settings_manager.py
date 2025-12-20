"""
UISettingsManager テスト

app/services/ui_settings_manager.py のテスト
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "app"))

# streamlitをモック
mock_st = MagicMock()
mock_st.session_state = {}
mock_st.cache_data = lambda **kwargs: lambda f: f
sys.modules['streamlit'] = mock_st

from services.ui_settings_manager import (
    TrainingAdvancedParams,
    SyntheticParams,
    DatasetPreparationParams,
    EvaluationParams,
    UISettings,
    UISettingsManager
)


# =============================================================================
# TestUISettingsManager
# =============================================================================


class TestUISettingsManager:
    """UISettingsManager クラスのテスト"""

    @pytest.fixture
    def mock_path_coordinator(self, tmp_path):
        """PathCoordinatorのモック"""
        coordinator = MagicMock()
        app_data = tmp_path / "app_data"
        app_data.mkdir(parents=True)
        coordinator.get_path.return_value = app_data
        return coordinator

    def test_initialization(self, mock_path_coordinator):
        """初期化テスト"""
        manager = UISettingsManager(path_coordinator=mock_path_coordinator)

        assert manager.path_coordinator == mock_path_coordinator

    def test_load_settings(self, mock_path_coordinator, tmp_path):
        """設定読み込みテスト"""
        # 設定ファイルを作成
        app_data = tmp_path / "app_data"
        settings_file = app_data / "ui_settings.json"
        settings_data = {
            "version": "1.0.0",
            "updated_at": "2024-01-01T00:00:00",
            "current_preset": "Fast",
            "training": {
                "advanced_params": {"epochs": 200},
                "synthetic": {"enabled": True}
            },
            "annotation": {
                "synthetic_generation": {},
                "dataset_preparation": {"validation_ratio": 0.3}
            },
            "evaluation": {"visual_conf": 0.5}
        }
        with open(settings_file, "w") as f:
            json.dump(settings_data, f)

        manager = UISettingsManager(path_coordinator=mock_path_coordinator)
        settings = manager.load()

        assert settings.current_preset == "Fast"
        assert settings.dataset_preparation.validation_ratio == 0.3

    def test_load_settings_file_not_found(self, mock_path_coordinator):
        """設定ファイル未発見テスト"""
        manager = UISettingsManager(path_coordinator=mock_path_coordinator)

        settings = manager.load()

        # デフォルト値が返される
        assert isinstance(settings, UISettings)
        assert settings.version == "1.0.0"

    def test_save_settings(self, mock_path_coordinator, tmp_path):
        """設定保存テスト"""
        manager = UISettingsManager(path_coordinator=mock_path_coordinator)

        settings = UISettings()
        settings.current_preset = "Custom"
        settings.training_params.epochs = 150

        manager.save(settings)

        # ファイルが作成されたことを確認
        app_data = tmp_path / "app_data"
        settings_file = app_data / "ui_settings.json"
        assert settings_file.exists()

        with open(settings_file) as f:
            saved = json.load(f)

        assert saved["current_preset"] == "Custom"

    def test_get_setting(self, mock_path_coordinator, tmp_path):
        """設定取得テスト"""
        # 設定ファイルを作成
        app_data = tmp_path / "app_data"
        settings_file = app_data / "ui_settings.json"
        settings_data = {
            "version": "1.0.0",
            "updated_at": "",
            "current_preset": "Competition",
            "training": {
                "advanced_params": {},
                "synthetic": {}
            },
            "annotation": {
                "synthetic_generation": {},
                "dataset_preparation": {}
            },
            "evaluation": {}
        }
        with open(settings_file, "w") as f:
            json.dump(settings_data, f)

        manager = UISettingsManager(path_coordinator=mock_path_coordinator)
        settings = manager.load()

        # 特定の設定値を取得
        assert settings.current_preset == "Competition"

    def test_get_setting_with_default(self, mock_path_coordinator):
        """デフォルト付き設定取得テスト"""
        manager = UISettingsManager(path_coordinator=mock_path_coordinator)
        settings = manager.load()

        # デフォルト値が正しく設定されている
        assert settings.training_params.lr0 is not None
        assert settings.training_params.optimizer is not None

    def test_set_setting(self, mock_path_coordinator, tmp_path):
        """設定設定テスト"""
        manager = UISettingsManager(path_coordinator=mock_path_coordinator)

        settings = manager.load()
        settings.current_preset = "Updated Preset"
        manager.save(settings)

        # 再読み込みして確認
        reloaded = manager.load()
        assert reloaded.current_preset == "Updated Preset"

    def test_delete_setting(self, mock_path_coordinator, tmp_path):
        """設定削除テスト（リセット）"""
        # 設定を保存
        manager = UISettingsManager(path_coordinator=mock_path_coordinator)
        settings = UISettings()
        settings.current_preset = "ToDelete"
        manager.save(settings)

        # 設定ファイルを削除してリセット
        app_data = tmp_path / "app_data"
        settings_file = app_data / "ui_settings.json"
        if settings_file.exists():
            settings_file.unlink()

        # デフォルト値が返される
        reloaded = manager.load()
        assert reloaded.current_preset == "Competition"  # デフォルト値

    def test_nested_settings(self, mock_path_coordinator, tmp_path):
        """ネストした設定テスト"""
        manager = UISettingsManager(path_coordinator=mock_path_coordinator)

        settings = UISettings()
        settings.training_params.lr0 = 0.005
        settings.training_synthetic.ratio = 3.0
        settings.dataset_preparation.validation_ratio = 0.25

        manager.save(settings)

        reloaded = manager.load()

        assert reloaded.training_params.lr0 == 0.005
        assert reloaded.training_synthetic.ratio == 3.0
        assert reloaded.dataset_preparation.validation_ratio == 0.25

    def test_settings_persistence(self, mock_path_coordinator, tmp_path):
        """設定の永続化テスト"""
        # 最初のマネージャで保存
        manager1 = UISettingsManager(path_coordinator=mock_path_coordinator)
        settings = UISettings()
        settings.current_preset = "Persistent"
        settings.evaluation.visual_conf = 0.75
        manager1.save(settings)

        # 新しいマネージャで読み込み
        manager2 = UISettingsManager(path_coordinator=mock_path_coordinator)
        loaded = manager2.load()

        assert loaded.current_preset == "Persistent"
        assert loaded.evaluation.visual_conf == 0.75


# =============================================================================
# TestDataclasses
# =============================================================================


class TestTrainingAdvancedParams:
    """TrainingAdvancedParams データクラスのテスト"""

    def test_default_values(self):
        """デフォルト値テスト"""
        params = TrainingAdvancedParams()

        assert params.hsv_h == 0.015
        assert params.lr0 == 0.001
        assert params.optimizer == "AdamW"
        assert params.amp is True

    def test_custom_values(self):
        """カスタム値テスト"""
        params = TrainingAdvancedParams(
            epochs=200,
            lr0=0.01,
            batch_size=32  # これはTrainingAdvancedParamsにはない可能性
        ) if hasattr(TrainingAdvancedParams, 'batch_size') else TrainingAdvancedParams(
            lr0=0.01
        )

        assert params.lr0 == 0.01


class TestSyntheticParams:
    """SyntheticParams データクラスのテスト"""

    def test_default_values(self):
        """デフォルト値テスト"""
        params = SyntheticParams()

        assert params.enabled is True
        assert params.ratio == 2.0
        assert params.max_objects == 3

    def test_custom_values(self):
        """カスタム値テスト"""
        params = SyntheticParams(
            enabled=False,
            ratio=1.5,
            max_objects=5
        )

        assert params.enabled is False
        assert params.ratio == 1.5
        assert params.max_objects == 5


class TestDatasetPreparationParams:
    """DatasetPreparationParams データクラスのテスト"""

    def test_default_values(self):
        """デフォルト値テスト"""
        params = DatasetPreparationParams()

        assert params.validation_ratio == 0.2
        assert params.group_continuous_frames is True
        assert params.group_interval_sec == 2.0


class TestEvaluationParams:
    """EvaluationParams データクラスのテスト"""

    def test_default_values(self):
        """デフォルト値テスト"""
        params = EvaluationParams()

        assert params.visual_conf == 0.25
        assert params.video_conf == 0.25
        assert params.robustness_conf == 0.25


class TestUISettings:
    """UISettings データクラスのテスト"""

    def test_default_values(self):
        """デフォルト値テスト"""
        settings = UISettings()

        assert settings.version == "1.0.0"
        assert settings.current_preset == "Competition"
        assert isinstance(settings.training_params, TrainingAdvancedParams)
        assert isinstance(settings.training_synthetic, SyntheticParams)
        assert isinstance(settings.dataset_preparation, DatasetPreparationParams)
        assert isinstance(settings.evaluation, EvaluationParams)

    def test_nested_access(self):
        """ネストしたアクセステスト"""
        settings = UISettings()

        # 訓練パラメータ
        assert settings.training_params.optimizer == "AdamW"

        # 合成パラメータ
        assert settings.training_synthetic.enabled is True

        # データセット準備
        assert settings.dataset_preparation.validation_ratio == 0.2
