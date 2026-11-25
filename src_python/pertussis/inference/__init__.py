"""Inference workflows for the core2026 transmission model.

The implementation is intentionally not imported eagerly.  This keeps
``python -m src_python.pertussis.inference.core2026`` free of the runpy
double-import warning and avoids loading the model when only the package
namespace is inspected.
"""

__all__ = ["Core2026Inference", "InferenceFit", "InferenceSettings", "run_inference"]


def __getattr__(name):
    if name in __all__:
        from .core2026 import Core2026Inference, InferenceFit, InferenceSettings, run_inference

        return {
            "Core2026Inference": Core2026Inference,
            "InferenceFit": InferenceFit,
            "InferenceSettings": InferenceSettings,
            "run_inference": run_inference,
        }[name]
    raise AttributeError(name)
