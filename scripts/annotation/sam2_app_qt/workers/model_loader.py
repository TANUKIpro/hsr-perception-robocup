"""
Model loader worker for async model loading.

Provides QThread-based worker for loading SAM2 models in the background.
"""

from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from sam2_app_qt.predictors.image_predictor import SAM2InteractivePredictor


class ModelLoaderWorker(QThread):
    """
    Worker thread for loading SAM2 model.

    Signals:
        progress: Emitted with status message during loading
        finished: Emitted when model is loaded successfully
        error: Emitted when model loading fails
    """

    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # SAM2InteractivePredictor
    error = pyqtSignal(str)

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        parent: Optional[object] = None,
    ):
        """
        Initialize model loader worker.

        Args:
            model_path: Path to SAM2 model checkpoint
            device: Device to run model on
            parent: Parent QObject
        """
        super().__init__(parent)
        self.model_path = model_path
        self.device = device

    def run(self) -> None:
        """Load the model in background thread."""
        try:
            predictor = SAM2InteractivePredictor(
                model_path=self.model_path,
                device=self.device,
            )

            def progress_callback(msg: str) -> None:
                self.progress.emit(msg)

            predictor.load_model(progress_callback=progress_callback)
            self.finished.emit(predictor)

        except Exception as e:
            self.error.emit(str(e))
