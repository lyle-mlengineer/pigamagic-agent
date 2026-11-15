"""Provider protocols and shared types for the agent.

Every external capability sits behind a narrow, typed protocol so the pipeline
can run offline against fixtures (Steps 1-7) and switch to real vendors later
(Steps 8, 9, 15) without touching pipeline code.

Rules for implementers (see base.py for the full contract):
- No vendor SDK object may escape an adapter. Map to the types here.
- Adapters own retries and timeouts; the pipeline only sees success or a
  typed AgentError.
- Cost is reported per call so the run layer can settle credits accurately.
"""

from pigamagic_agent.agent.providers.base import (
    AgentError,
    CostEstimate,
    GeneratedImage,
    ImageProvider,
    LLMProvider,
    LLMResult,
    OfflineModeError,
    ProviderError,
    ProviderNotConfigured,
    SearchHit,
    VisionProvider,
    VisionResult,
    WebImage,
    WebPage,
    WebProvider,
)

__all__ = [
    "AgentError",
    "CostEstimate",
    "GeneratedImage",
    "ImageProvider",
    "LLMProvider",
    "LLMResult",
    "OfflineModeError",
    "ProviderError",
    "ProviderNotConfigured",
    "SearchHit",
    "VisionProvider",
    "VisionResult",
    "WebImage",
    "WebPage",
    "WebProvider",
]
