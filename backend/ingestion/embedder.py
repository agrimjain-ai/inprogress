import logging
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    def __init__(
        self, 
        model_name: str = "all-mpnet-base-v2", 
        device: Optional[str] = None,
        lazy_load: bool = False
    ):
        """
        Initializes the document embedder with automatic device fallback,
        mixed-precision optimizations, and eager warm-up capability.
        """
        self.model_name = model_name
        self._model = None
        self._lock = threading.RLock()
        
        # Automatic hardware detection (CUDA vs. CPU)
        if not device:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        if not lazy_load:
            # Trigger eager loading and warm-up during initialization
            _ = self.model

    @property
    def model(self):
        """
        Thread-safe getter that loads, optimizes, warms up, and caches the model.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info(
                        "Initializing SentenceTransformer '%s' on device '%s'...", 
                        self.model_name, self.device
                    )
                    try:
                        import torch
                        from sentence_transformers import SentenceTransformer
                        
                        # Optimization: Use float16 on CUDA to halve memory and boost throughput
                        model_kwargs = {}
                        if self.device == "cuda":
                            model_kwargs["torch_dtype"] = torch.float16
                            logger.info("Using FP16 mixed-precision for CUDA inference.")

                        loaded_model = SentenceTransformer(
                            self.model_name, 
                            device=self.device, 
                            **model_kwargs
                        )
                        
                        # Optimization: Warm up CUDA kernels to eliminate first-request latency spike
                        logger.info("Warming up model with a dummy inference pass...")
                        loaded_model.encode(["warmup"], show_progress_bar=False)
                        
                        self._model = loaded_model
                        logger.info("Model initialization and warm-up complete.")
                        
                    except Exception as e:
                        logger.error(
                            "Failed to load SentenceTransformer model %s: %s", 
                            self.model_name, e, exc_info=True
                        )
                        raise RuntimeError(f"Could not initialize embedding model: {e}") from e
        return self._model

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates dense vector embeddings from a list of text segments.
        
        Args:
            texts: List of strings to be embedded.
            batch_size: Number of chunks encoded simultaneously.
            
        Returns:
            A list of vector embeddings.
        """
        if not texts:
            return []

        # Validate input types before processing to avoid downstream failures
        cleaned_texts: List[str] = []
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValueError(f"Input at index {i} is not a string: {type(t)}")
            
            stripped = t.strip()
            if not stripped:
                logger.warning("Empty or whitespace-only string detected at index %d.", i)
            cleaned_texts.append(stripped)

        try:
            # Perform batch inference
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Convert NumPy array output to a native Python list of floats
            return embeddings.tolist()
            
        except Exception as e:
            logger.exception("Inference failed during batch embedding generation.")
            raise RuntimeError(f"Inference run failed: {e}") from e