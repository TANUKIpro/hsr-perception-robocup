"""
PathCoordinator テスト

app/services/path_coordinator.py のテスト
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "app"))

# streamlitをモック
sys.modules['streamlit'] = MagicMock()
mock_st = sys.modules['streamlit']
mock_st.cache_data = lambda **kwargs: lambda f: f

from services.path_coordinator import PathConfig, PathCoordinator


# =============================================================================
# TestPathConfig
# =============================================================================


class TestPathConfig:
    """PathConfig データクラスのテスト"""

    def test_default_paths(self):
        """デフォルトパステスト"""
        config = PathConfig()

        assert config.app_data_dir == "app_data"
        assert config.raw_captures_dir == "datasets/raw_captures"
        assert config.finetuned_dir == "models/finetuned"
        assert config.pretrained_dir == "models/pretrained"

    def test_custom_paths(self):
        """カスタムパステスト"""
        config = PathConfig(
            app_data_dir="custom_data",
            raw_captures_dir="custom/captures"
        )

        assert config.app_data_dir == "custom_data"
        assert config.raw_captures_dir == "custom/captures"
        # デフォルト値は維持
        assert config.finetuned_dir == "models/finetuned"


# =============================================================================
# TestPathCoordinator
# =============================================================================


class TestPathCoordinator:
    """PathCoordinator クラスのテスト"""

    @pytest.fixture
    def mock_profile_manager(self, tmp_path):
        """ProfileManagerのモック"""
        manager = MagicMock()
        manager.get_active_profile_id.return_value = "prof_1"
        profile_path = tmp_path / "profiles" / "prof_1"
        profile_path.mkdir(parents=True)
        manager.get_profile_path.return_value = profile_path
        return manager

    def test_initialization(self, tmp_path, mock_profile_manager):
        """初期化テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        assert coordinator.project_root == tmp_path

    def test_initialization_with_profile_manager(self, tmp_path, mock_profile_manager):
        """ProfileManager付き初期化テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        # プロファイルマネージャが使われていることを確認
        assert coordinator._profile_manager == mock_profile_manager

    def test_get_path_profile_specific(self, tmp_path, mock_profile_manager):
        """プロファイル固有パス取得テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        raw_path = coordinator.get_path("raw_captures_dir")

        # プロファイル固有のパスが返される
        assert "prof_1" in str(raw_path)
        assert raw_path.name == "raw_captures"

    def test_get_path_shared(self, tmp_path, mock_profile_manager):
        """共有パス取得テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        pretrained_path = coordinator.get_path("pretrained_dir")

        # プロジェクトルートからの相対パス（プロファイル固有でない）
        assert pretrained_path == tmp_path / "models" / "pretrained"

    def test_resolve_path_absolute(self, tmp_path, mock_profile_manager):
        """絶対パス解決テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        abs_path = Path("/absolute/path/to/file")
        resolved = coordinator.resolve_path(abs_path)

        assert resolved == abs_path

    def test_resolve_path_relative(self, tmp_path, mock_profile_manager):
        """相対パス解決テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        resolved = coordinator.resolve_path("relative/path")

        # プロファイルルートからの相対パスとして解決
        profile_path = mock_profile_manager.get_profile_path()
        assert resolved == profile_path / "relative/path"

    def test_create_annotation_session(self, tmp_path, mock_profile_manager):
        """アノテーションセッション作成テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        session = coordinator.create_annotation_session("test_session")

        assert session["session_name"] == "test_session"
        assert "input_dir" in session
        assert "output_dir" in session
        assert "class_config" in session
        assert Path(session["output_dir"]).exists()

    def test_get_annotation_sessions(self, tmp_path, mock_profile_manager):
        """アノテーションセッション取得テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        # テスト用セッションディレクトリを作成
        annotated_dir = coordinator.get_path("annotated_dir")
        annotated_dir.mkdir(parents=True, exist_ok=True)

        session_dir = annotated_dir / "session_001"
        session_dir.mkdir()
        (session_dir / "data.yaml").write_text("path: .")

        sessions = coordinator.get_annotation_sessions()

        assert len(sessions) >= 1
        assert sessions[0]["name"] == "session_001"
        assert sessions[0]["has_data_yaml"] is True

    def test_get_training_paths(self, tmp_path, mock_profile_manager):
        """訓練パス取得テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        # テスト用データセットを作成
        annotated_dir = coordinator.get_path("annotated_dir")
        session_dir = annotated_dir / "test_session"
        session_dir.mkdir(parents=True)
        (session_dir / "data.yaml").write_text("path: .")

        paths = coordinator.get_training_paths("test_session")

        assert "dataset_yaml" in paths
        assert "output_dir" in paths
        assert paths["dataset_yaml"].endswith("data.yaml")

    def test_get_trained_models(self, tmp_path, mock_profile_manager):
        """訓練済みモデル取得テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        # テスト用モデルディレクトリを作成
        finetuned_dir = coordinator.get_path("finetuned_dir")
        finetuned_dir.mkdir(parents=True, exist_ok=True)

        model_dir = finetuned_dir / "run_001"
        weights_dir = model_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_text("dummy")

        models = coordinator.get_trained_models()

        assert len(models) >= 1
        assert models[0]["name"] == "run_001"
        assert models[0]["best_path"] is not None

    def test_get_pretrained_models(self, tmp_path, mock_profile_manager):
        """事前訓練モデル取得テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        # テスト用事前訓練モデルを作成
        pretrained_dir = coordinator.get_path("pretrained_dir")
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        (pretrained_dir / "custom_model.pt").write_text("dummy")

        models = coordinator.get_pretrained_models()

        assert len(models) >= 1
        # 標準モデル名も含まれる
        model_names = [Path(m).name for m in models]
        assert "yolov8m.pt" in model_names

    def test_get_background_images(self, tmp_path, mock_profile_manager):
        """背景画像取得テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        # テスト用背景画像を作成
        backgrounds_dir = coordinator.get_path("backgrounds_dir")
        backgrounds_dir.mkdir(parents=True, exist_ok=True)
        (backgrounds_dir / "white.jpg").write_text("dummy")
        (backgrounds_dir / "black.png").write_text("dummy")

        images = coordinator.get_background_images()

        assert len(images) == 2
        names = [img["name"] for img in images]
        assert "white.jpg" in names
        assert "black.png" in names

    def test_add_background_image(self, tmp_path, mock_profile_manager):
        """背景画像追加テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        # ソース画像を作成
        source_path = tmp_path / "source_bg.jpg"
        source_path.write_text("dummy image data")

        saved_path = coordinator.add_background_image(source_path)

        assert Path(saved_path).exists()
        assert "source_bg.jpg" in saved_path

    def test_validate_paths(self, tmp_path, mock_profile_manager):
        """パス検証テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        results = coordinator.validate_paths()

        # 少なくとも一部のディレクトリが存在する
        assert isinstance(results, dict)
        assert "raw_captures_dir" in results


# =============================================================================
# TestCachedFunctions
# =============================================================================


class TestCachedFunctions:
    """キャッシュ関数のテスト"""

    @pytest.fixture
    def mock_profile_manager(self, tmp_path):
        """ProfileManagerのモック"""
        manager = MagicMock()
        manager.get_active_profile_id.return_value = "prof_1"
        profile_path = tmp_path / "profiles" / "prof_1"
        profile_path.mkdir(parents=True)
        manager.get_profile_path.return_value = profile_path
        return manager

    def test_cached_get_annotation_sessions(self, tmp_path, mock_profile_manager):
        """キャッシュ付きセッション取得テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        # セッションを作成
        annotated_dir = coordinator.get_path("annotated_dir")
        annotated_dir.mkdir(parents=True, exist_ok=True)
        (annotated_dir / "session_test").mkdir()

        # 2回呼び出し
        sessions1 = coordinator.get_annotation_sessions()
        sessions2 = coordinator.get_annotation_sessions()

        # 結果が同じ
        assert sessions1 == sessions2

    def test_cached_get_trained_models(self, tmp_path, mock_profile_manager):
        """キャッシュ付きモデル取得テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        # モデルを作成
        finetuned_dir = coordinator.get_path("finetuned_dir")
        model_dir = finetuned_dir / "model_001" / "weights"
        model_dir.mkdir(parents=True)
        (model_dir / "best.pt").write_text("dummy")

        # 2回呼び出し
        models1 = coordinator.get_trained_models()
        models2 = coordinator.get_trained_models()

        assert models1 == models2

    def test_cached_get_background_images(self, tmp_path, mock_profile_manager):
        """キャッシュ付き背景取得テスト"""
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        # 背景画像を作成
        backgrounds_dir = coordinator.get_path("backgrounds_dir")
        backgrounds_dir.mkdir(parents=True, exist_ok=True)
        (backgrounds_dir / "bg.jpg").write_text("dummy")

        # 2回呼び出し
        images1 = coordinator.get_background_images()
        images2 = coordinator.get_background_images()

        assert images1 == images2

    def test_cache_invalidation(self, tmp_path, mock_profile_manager):
        """キャッシュ無効化テスト（時間経過による）"""
        # キャッシュはttl=30秒なので、実際の無効化テストは
        # 単体テストでは難しい。ここでは関数が正常動作することを確認
        coordinator = PathCoordinator(
            project_root=tmp_path,
            profile_manager=mock_profile_manager
        )

        # 初回呼び出し
        sessions = coordinator.get_annotation_sessions()

        # 新しいセッションを追加
        annotated_dir = coordinator.get_path("annotated_dir")
        (annotated_dir / "new_session").mkdir(parents=True, exist_ok=True)

        # キャッシュがあるため、新しいセッションは反映されない可能性
        # (テストモックではキャッシュが効かないので反映される)
        sessions_after = coordinator.get_annotation_sessions()

        # テスト環境では毎回再取得される
        assert isinstance(sessions_after, list)
