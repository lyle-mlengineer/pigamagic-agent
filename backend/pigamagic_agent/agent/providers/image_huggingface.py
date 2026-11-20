"""Local Hugging Face provider (diffusers), for Colab testing.

Why this adapter exists: it lets the whole render pipeline run on a free Colab
GPU against **real** open-weight models, with no API key, no per-image cost, and
no vendor account. It is the cheapest honest way to answer the one question no
fixture can: *does the generated person look like the person in the references?*

It satisfies the same `ImageProvider` protocol as Gemini/OpenAI/NVIDIA, so
`render_plan()` cannot tell the difference.

## Model registry

Model behaviour differs enough that guessing parameters produces garbage, so each
family is described once in `MODEL_SPECS`:

| model | identity refs | notes |
|---|---|---|
| `Qwen/Qwen-Image-2.1-Turbo` | **yes, up to 3** | ungated, 8 steps, text-to-image **and** multi-reference editing. **The default.** |
| `Qwen/Qwen-Image-Edit-2509` | **yes, up to 3** | documented to preserve facial identity across poses |
| `Qwen/Qwen-Image-Edit` | yes, 1 | older, weaker identity retention |
| `Qwen/Qwen-Image` | no | strongest text rendering; text-to-image only |
| `black-forest-labs/FLUX.1-Kontext-dev` | yes, 1 | gated; instruction-based editing |
| `black-forest-labs/FLUX.1-schnell` | no | 4 steps, fastest; **gated since 2026**, needs a free token |
| `black-forest-labs/FLUX.1-dev` | no | gated, higher quality text-to-image |
| `stabilityai/stable-diffusion-3.5-large` | no | gated, well-supported fallback |
| `stabilityai/sdxl-turbo` | no | ungated, ~2.6 GB, the quick plumbing check |

## Hugging Face gating (the one thing that breaks a fresh Colab)

A checkpoint is *gated* when the HF API reports `gated: "auto"`: it downloads
only for an account that has accepted the licence on the model page, with a
token attached. This changes without notice — `FLUX.1-schnell` was ungated when
this module was written and is not any more — so the flags live in
`MODEL_SPECS`, `preflight()` refuses early instead of after a 40 GB download,
and `UNGATED_MODELS` is the honest list of what still needs no account.

## Three API traps this module handles

1. **Qwen ignores `guidance_scale`.** Real guidance is `true_cfg_scale`, and CFG
   only activates when a `negative_prompt` is supplied. Passing `guidance_scale`
   silently does nothing.
2. **`QwenImage21Pipeline` has no `guidance_scale` and no `max_sequence_length`
   at all** — it rejects them. It samples guidance-free (`true_cfg_scale=1.0`),
   ships its own 8-step sigma schedule in the checkpoint (so
   `num_inference_steps` is ignored), and takes a *list* of condition images.
   Everything else in this table is passed only when the spec says so.
3. **FLUX.1-schnell is timestep-distilled**: it requires `guidance_scale=0.0` and
   `max_sequence_length <= 256`. Passing the dev defaults produces mush.

Parameters therefore come from the spec table, never from hardcoded guesses.

`Qwen-Image-2.1-Turbo` needs a diffusers build that has `QwenImage21Pipeline`
and `transformers>=5.17` for the `Qwen3VL` text encoder:

```bash
pip install -q git+https://github.com/huggingface/diffusers.git "transformers>=5.17" accelerate safetensors
```

Dependencies are imported lazily, so `pytest` on a laptop stays fast and does not
need torch/diffusers installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from pigamagic_agent.agent.providers.base import CostEstimate, GeneratedImage, ProviderError
from pigamagic_agent.agent.providers.http_base import require_positive, sniff_image_type
from pigamagic_agent.agent.providers.image_base import BaseImageProvider, ImageCapabilities


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """How to run one checkpoint correctly."""

    model_id: str
    family: str  # "qwen" | "flux" | "sd"
    identity_refs: int = 0  # 0 = text-to-image only
    gated: bool = False
    steps: int = 28
    guidance_scale: float | None = 3.5
    true_cfg_scale: float | None = None  # Qwen's real guidance knob
    max_sequence_length: int | None = 512
    needs_negative_prompt: bool = False
    # False when the pipeline has no CFG to feed a negative prompt into, so the
    # caller's negatives are folded into the positive prompt instead of ignored.
    supports_negative_prompt: bool = True
    use_kv_cache: bool | None = None  # Qwen-Image-2.1 pipelines only
    dtype: str | None = None  # preferred torch dtype; None = fp16 on GPU
    # "auto" offloads when the checkpoint is in FP16_MODELS, "sequential" keeps
    # one module on the GPU at a time (needed when a single component, e.g. a
    # 17 GB text encoder, is larger than the card), "none" just .to(device).
    offload: str = "auto"
    sizes: tuple[tuple[int, int], ...] = ((1024, 1024),)
    note: str = ""


_SQUARE_AND_LANDSCAPE: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1344, 768),
    (768, 1344),
    (1184, 888),
    (888, 1184),
    (512, 512),
)

# Qwen-Image-2.1 documents 2 MP presets (2048x2048, 2400x1792, 2752x1536, ...).
# Those are 4x the latent tokens of a 1 MP frame, so the table below is the same
# ratios at 1 MP: the pipeline snaps both sides to the VAE's 16-pixel grid, and a
# free T4 can actually finish a shot list.
_QWEN21_SIZES: tuple[tuple[int, int], ...] = (
    (1024, 1024),  # 1:1
    (1152, 864),  # 4:3
    (864, 1152),  # 3:4
    (1248, 832),  # 3:2
    (832, 1248),  # 2:3
    (1360, 768),  # 16:9
    (768, 1360),  # 9:16
)


MODEL_SPECS: dict[str, ModelSpec] = {
    # --- Qwen: the identity-preserving choice --------------------------
    "Qwen/Qwen-Image-2.1-Turbo": ModelSpec(
        model_id="Qwen/Qwen-Image-2.1-Turbo",
        family="qwen",
        identity_refs=3,  # the card shows multi-reference composition
        steps=8,  # fixed 8-step sigma schedule saved with the checkpoint
        # QwenImage21Pipeline has neither kwarg: guidance is true_cfg_scale, and
        # the checkpoint is tuned for guidance-free sampling (true_cfg_scale=1.0).
        guidance_scale=None,
        true_cfg_scale=None,
        needs_negative_prompt=False,
        supports_negative_prompt=False,  # no CFG, so fold negatives into the prompt
        max_sequence_length=None,
        use_kv_cache=True,
        dtype="bfloat16",
        # 17.5 GB text encoder + 14 GB transformer cannot both sit on a 16 GB T4.
        offload="sequential",
        sizes=_QWEN21_SIZES,
        note="ungated, 8 steps; text-to-image and multi-reference editing",
    ),
    "Qwen/Qwen-Image-Edit-2509": ModelSpec(
        model_id="Qwen/Qwen-Image-Edit-2509",
        family="qwen",
        identity_refs=3,
        steps=40,
        guidance_scale=1.0,
        true_cfg_scale=4.0,
        needs_negative_prompt=True,
        sizes=_SQUARE_AND_LANDSCAPE,
        note="preserves facial identity across poses; best for this app",
    ),
    "Qwen/Qwen-Image-Edit": ModelSpec(
        model_id="Qwen/Qwen-Image-Edit",
        family="qwen",
        identity_refs=1,
        steps=40,
        guidance_scale=1.0,
        true_cfg_scale=4.0,
        needs_negative_prompt=True,
        sizes=_SQUARE_AND_LANDSCAPE,
        note="single-reference editing",
    ),
    "Qwen/Qwen-Image": ModelSpec(
        model_id="Qwen/Qwen-Image",
        family="qwen",
        identity_refs=0,
        steps=50,
        guidance_scale=None,
        true_cfg_scale=4.0,
        needs_negative_prompt=True,
        sizes=_SQUARE_AND_LANDSCAPE,
        note="excellent text rendering; text-to-image only",
    ),
    # --- FLUX ----------------------------------------------------------
    "black-forest-labs/FLUX.1-Kontext-dev": ModelSpec(
        model_id="black-forest-labs/FLUX.1-Kontext-dev",
        family="flux",
        identity_refs=1,
        gated=True,
        steps=28,
        guidance_scale=2.5,
        max_sequence_length=512,
        sizes=_SQUARE_AND_LANDSCAPE,
        note="instruction-based editing; gated",
    ),
    "black-forest-labs/FLUX.1-schnell": ModelSpec(
        model_id="black-forest-labs/FLUX.1-schnell",
        family="flux",
        identity_refs=0,
        gated=True,  # was ungated; Hugging Face flipped it to gated:auto
        steps=4,
        guidance_scale=0.0,  # timestep-distilled: must be 0
        max_sequence_length=256,  # documented ceiling for schnell
        sizes=_SQUARE_AND_LANDSCAPE,
        note="gated (auto-approve): needs a free HF token; 4 steps, fastest",
    ),
    "black-forest-labs/FLUX.1-dev": ModelSpec(
        model_id="black-forest-labs/FLUX.1-dev",
        family="flux",
        identity_refs=0,
        gated=True,
        steps=28,
        guidance_scale=3.5,
        max_sequence_length=512,
        sizes=_SQUARE_AND_LANDSCAPE,
        note="higher quality text-to-image; gated",
    ),
    # --- Stable Diffusion ----------------------------------------------
    "stabilityai/stable-diffusion-3.5-large": ModelSpec(
        model_id="stabilityai/stable-diffusion-3.5-large",
        family="sd",
        identity_refs=0,
        gated=True,
        steps=40,
        guidance_scale=4.5,
        sizes=_SQUARE_AND_LANDSCAPE,
        note="gated; well-supported fallback",
    ),
    "stabilityai/sdxl-turbo": ModelSpec(
        model_id="stabilityai/sdxl-turbo",
        family="sd",
        identity_refs=0,
        steps=2,
        guidance_scale=0.0,  # distilled: CFG must stay off
        max_sequence_length=None,  # no T5 encoder to truncate
        sizes=((512, 512), (768, 768), (1024, 1024)),
        note="ungated and tiny (~2.6 GB); the zero-setup smoke test",
    ),
}

# Ungated, 8 steps, and the only default that both keeps a likeness and needs
# no account: see `gated_help()` for the checkpoints that do.
DEFAULT_MODEL = "Qwen/Qwen-Image-2.1-Turbo"

# Small, ungated, ~2.6 GB: for checking the plumbing without a 32 GB download.
QUICK_TEST_MODEL = "stabilityai/sdxl-turbo"

# Checkpoints that download with no account, no licence click and no token.
# Verified against https://huggingface.co/api/models/<id> (`gated: false`);
# keep this in sync with MODEL_SPECS[*].gated when Hugging Face changes its mind.
UNGATED_MODELS: tuple[str, ...] = (
    "Qwen/Qwen-Image-2.1-Turbo",
    "Qwen/Qwen-Image-Edit-2509",
    "Qwen/Qwen-Image-Edit",
    "Qwen/Qwen-Image",
    "stabilityai/sdxl-turbo",
)

# Weights ~12-20 GB for FLUX/Qwen; a free Colab T4 has 16 GB, so offloading and
# fp16 are required rather than nice-to-have.
FP16_MODELS = frozenset(
    {
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-Edit",
        "Qwen/Qwen-Image-Edit-2509",
        "Qwen/Qwen-Image-2.1-Turbo",
    }
)


def resolve_spec(model_id: str) -> ModelSpec:
    """Look up a spec, or build a cautious default for an unknown checkpoint."""
    if model_id in MODEL_SPECS:
        return MODEL_SPECS[model_id]
    return ModelSpec(
        model_id=model_id,
        family=_guess_family(model_id),
        identity_refs=0,
        steps=28,
        guidance_scale=3.5,
        sizes=_SQUARE_AND_LANDSCAPE,
        note="unknown checkpoint; assuming text-to-image",
    )


def _guess_family(model_id: str) -> str:
    lowered = model_id.casefold()
    if "qwen" in lowered:
        return "qwen"
    if "flux" in lowered:
        return "flux"
    return "sd"


def is_gated(model_id: str) -> bool:
    """True when the checkpoint needs a licence click and a token to download."""
    return resolve_spec(model_id).gated


def ungated_alternatives(model_id: str, *, limit: int = 2) -> list[str]:
    """Ungated checkpoints to switch to, best-first, never the failing one."""
    return [m for m in UNGATED_MODELS if m != model_id][:limit]


def gated_help(model_id: str) -> str:
    """The two clicks that fix a gated download, plus the way around it.

    Written as one string so the same text reaches a Colab user, a CLI error and
    a log line. It must never recommend the checkpoint that just failed.
    """
    text = (
        f"{model_id} is gated on Hugging Face: it downloads only for an account "
        f"that has accepted its licence, using a token.\n"
        f"  1. accept the licence: https://huggingface.co/{model_id}\n"
        f"  2. copy a read token: https://huggingface.co/settings/tokens\n"
        f"  3. pass it as AGENT_IMAGE_API_KEY (HF_TOKEN for scripts/colab_smoke.py)\n"
        f"Approval is automatic, so a fresh token works immediately."
    )
    alternatives = ungated_alternatives(model_id)
    if alternatives:
        text += (
            "\nNo account? These checkpoints are ungated and download as-is: "
            + ", ".join(alternatives)
            + "."
        )
    # Trailing newline: ProviderError.__str__ appends "provider=... retryable=..."
    # on the same line, and it must not run into the last step.
    return text + "\n"


class HuggingFaceImageProvider(BaseImageProvider):
    """Runs a diffusion checkpoint locally, in-process."""

    name: str = "huggingface"
    capabilities = ImageCapabilities(
        supports_identity_refs=False,
        max_identity_refs=0,
        supports_negative_prompt=True,
        supports_seed=True,
        supports_structural_ref=False,
        supported_sizes=_SQUARE_AND_LANDSCAPE,
    )
    price_per_image_usd = 0.0

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "",
        settings: Any = None,
        budget: Any = None,
        device: str | None = None,
        dtype: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Skip HttpProvider.__post_init__: local runs need no client or budget.
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.settings = settings
        self.budget = budget
        self.base_url = ""
        self.timeout_s = 0.0
        self.max_attempts = 1
        self._client = None
        self.device = device
        self.dtype = dtype
        self.pipeline = None
        self._loaded_model = ""
        self._apply_spec()

    # -- configuration ---------------------------------------------------

    def _apply_spec(self) -> None:
        """Derive capabilities from the checkpoint's real abilities."""
        spec = resolve_spec(self.model)
        self.spec = spec
        self.capabilities = ImageCapabilities(
            supports_identity_refs=spec.identity_refs > 0,
            max_identity_refs=spec.identity_refs,
            supports_negative_prompt=spec.supports_negative_prompt,
            supports_seed=True,
            supports_structural_ref=spec.identity_refs > 0,
            supported_sizes=spec.sizes,
        )

    def require_credentials(self) -> None:
        return None

    def guard_offline(self) -> None:
        return None

    def resolve_model(self, override: str | None) -> str:
        return override or self.model or DEFAULT_MODEL

    def preflight(self, model: str | None = None) -> None:
        """Refuse a gated checkpoint *before* the multi-gigabyte download.

        Without this, a tokenless Colab run spends minutes fetching nothing and
        then fails with a 403 that reads like a network fault.
        """
        target = self.resolve_model(model)
        if is_gated(target) and not self.api_key:
            raise ProviderError(gated_help(target), provider=self.name, retryable=False)

    # -- loading ---------------------------------------------------------

    def load(self, model: str | None = None) -> Any:
        target = self.resolve_model(model)
        if self.pipeline is not None and self._loaded_model == target:
            return self.pipeline

        try:
            import torch  # noqa: PLC0415
            from diffusers import DiffusionPipeline  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ProviderError(
                "The huggingface provider needs torch and diffusers. In Colab run:\n"
                "  !pip install -q git+https://github.com/huggingface/diffusers.git "
                "'transformers>=5.17' accelerate safetensors",
                provider=self.name,
                retryable=False,
            ) from exc

        # Cheap, local, and worth doing before touching the network.
        self.preflight(target)

        spec = resolve_spec(target)
        has_cuda = self._has_cuda(torch)

        # A spec may name the dtype its authors validated (Qwen-Image-2.1 asks
        # for bfloat16), but a T4 is sm_75 and has no native bf16, so fall back
        # instead of failing late inside a matmul.
        preferred = self.dtype or spec.dtype
        if preferred:
            torch_dtype = getattr(torch, preferred, torch.float32)
            if preferred == "bfloat16" and not self._supports_bfloat16(torch):
                import warnings  # noqa: PLC0415

                warnings.warn(
                    f"{target} prefers bfloat16, which this GPU cannot run; "
                    f"using float16. Checkpoints may differ slightly in quality.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                torch_dtype = getattr(torch, "float16", torch_dtype)
        elif has_cuda:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        try:
            pipeline = DiffusionPipeline.from_pretrained(
                target,
                torch_dtype=torch_dtype,
                token=self.api_key or None,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            lowered = message.casefold()
            # GatedRepoError first: its text also contains 401/403, so checking
            # the status codes first would blame the token for a licence issue.
            if "gated" in lowered or "awaiting a review" in lowered:
                raise ProviderError(
                    gated_help(target), provider=self.name, retryable=False
                ) from exc
            if "401" in lowered or "unauthorized" in lowered or "invalid credentials" in lowered:
                raise ProviderError(
                    f"Hugging Face rejected the token while loading {target}. "
                    f"Create a read token at https://huggingface.co/settings/tokens "
                    f"and pass it as AGENT_IMAGE_API_KEY with no quotes and no "
                    f"'Bearer ' prefix.",
                    provider=self.name,
                    retryable=False,
                ) from exc
            if "404" in lowered or "not found" in lowered or "repo_id" in lowered:
                raise ProviderError(
                    f"{target} is not a valid Hugging Face repo id. Use "
                    f"'organisation/model'. Ungated options: "
                    + ", ".join(UNGATED_MODELS),
                    provider=self.name,
                    retryable=False,
                ) from exc
            if "403" in lowered:
                raise ProviderError(
                    gated_help(target), provider=self.name, retryable=False
                ) from exc
            raise ProviderError(
                f"Could not load {target}: {message[:300]}",
                provider=self.name,
                retryable=False,
            ) from exc

        # Colab T4 has 16 GB; offload rather than OOM.
        device = self.device or ("cuda" if has_cuda else "cpu")
        try:
            if has_cuda and spec.offload == "sequential":
                # One module on the card at a time: slower, but the only way the
                # 17.5 GB Qwen3-VL text encoder fits beside a 14 GB transformer.
                pipeline.enable_sequential_cpu_offload()
            elif has_cuda and (spec.offload == "model" or target in FP16_MODELS):
                pipeline.enable_model_cpu_offload()
            else:
                pipeline = pipeline.to(device)
        except Exception:  # noqa: BLE001
            try:
                pipeline = pipeline.to("cpu")
            except Exception:  # noqa: BLE001
                pass

        for method in ("enable_attention_slicing", "enable_vae_slicing"):
            if hasattr(pipeline, method):
                try:
                    getattr(pipeline, method)()
                except Exception:  # noqa: BLE001
                    pass

        self.pipeline = pipeline
        self._loaded_model = target
        return pipeline

    @staticmethod
    def _has_cuda(torch_module: Any) -> bool:
        try:
            return bool(torch_module.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _supports_bfloat16(torch_module: Any) -> bool:
        """bf16 needs Ampere or newer (sm_80); a free T4 is sm_75."""
        try:
            return bool(torch_module.cuda.is_available()) and (
                int(torch_module.cuda.get_device_capability(0)[0]) >= 8
            )
        except Exception:  # noqa: BLE001
            return False


    @staticmethod
    def _dependencies_available() -> bool:
        """True when torch and diffusers can be imported."""
        import importlib.util  # noqa: PLC0415

        return all(
            importlib.util.find_spec(name) is not None for name in ("torch", "diffusers")
        )

    # -- prompt adapters -------------------------------------------------

    def adapt_prompt(self, prompt: str, *, model: str | None = None) -> str:
        """Fit a prompt to the checkpoint's text-encoder limits."""
        target = self.resolve_model(model)
        spec = resolve_spec(target)
        text = prompt.strip()
        if spec.max_sequence_length is not None and spec.max_sequence_length <= 256:
            # schnell truncates at 256 tokens; drop the reference block first so
            # the subject and action survive.
            from pigamagic_agent.agent.prompts import REFERENCE_END, REFERENCE_START

            if REFERENCE_START in text and REFERENCE_END in text:
                start = text.index(REFERENCE_START)
                end = text.index(REFERENCE_END) + len(REFERENCE_END)
                text = f"{text[:start]}{text[end:]}".strip()
        return text

    # -- generation ------------------------------------------------------

    def generate_sync(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        identity_refs: dict[str, list[bytes]] | None = None,
        structural_ref: bytes | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> GeneratedImage:
        """Blocking generation. Call through `generate` from async code."""
        if not prompt or not prompt.strip():
            raise ProviderError("prompt must not be empty", provider=self.name, retryable=False)
        require_positive(width, "width")
        require_positive(height, "height")

        target = self.resolve_model(model)
        spec = resolve_spec(target)

        ref_bytes = self._collect_refs(identity_refs, structural_ref)
        # Guard on the INPUT, not the capped list: _collect_refs returns [] when
        # the spec has no identity support, which would silently drop refs and
        # produce a stranger instead of a clear error.
        requested_refs = sum(len(v) for v in (identity_refs or {}).values()) + (
            1 if structural_ref else 0
        )
        if requested_refs and spec.identity_refs <= 0:
            raise ProviderError(
                f"{target} is a text-to-image checkpoint and cannot keep a real "
                f"person's likeness. Use 'Qwen/Qwen-Image-Edit-2509' (documented "
                f"to preserve facial identity) for identity-preserving runs.",
                provider=self.name,
                retryable=False,
            )

        # Check the free, local preconditions BEFORE loading ~20 GB of weights.
        if not self._dependencies_available():
            raise ProviderError(
                "The huggingface provider needs torch and diffusers. In Colab run: "
                "pip install -q git+https://github.com/huggingface/diffusers.git "
                "'transformers>=5.17' accelerate safetensors",
                provider=self.name,
                retryable=False,
            )

        pipeline = self.load(target)
        params = dict(params or {})

        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        # A guidance-free checkpoint (Qwen-Image-2.1) has no CFG to feed a
        # negative prompt into, and the pipeline silently ignores one. Fold the
        # caller's negatives into the positive prompt instead of dropping them.
        if not spec.supports_negative_prompt and negative_prompt.strip():
            prompt = self.compose_prompt(prompt, negative_prompt)
            negative_prompt = ""

        generator = torch.Generator(device="cpu")
        # Qwen's docs use torch.manual_seed; a CPU generator is reproducible and
        # avoids device-specific seed drift.
        generator.manual_seed(int(seed) if seed is not None else 0)

        call_kwargs: dict[str, Any] = {
            "prompt": self.adapt_prompt(prompt, model=target),
            "width": width,
            "height": height,
            "generator": generator,
            "num_inference_steps": int(params.get("steps", spec.steps)),
        }

        # Only pass what this pipeline actually accepts: QwenImage21Pipeline
        # rejects guidance_scale and max_sequence_length outright.
        if spec.use_kv_cache is not None:
            call_kwargs["use_kv_cache"] = bool(
                params.get("use_kv_cache", spec.use_kv_cache)
            )

        if spec.guidance_scale is not None:
            call_kwargs["guidance_scale"] = float(
                params.get("guidance_scale", spec.guidance_scale)
            )
        if spec.max_sequence_length is not None:
            call_kwargs["max_sequence_length"] = int(
                params.get("max_sequence_length", spec.max_sequence_length)
            )

        # Qwen uses true_cfg_scale and needs a negative prompt to enable CFG.
        if spec.true_cfg_scale is not None:
            call_kwargs["true_cfg_scale"] = float(
                params.get("true_cfg_scale", spec.true_cfg_scale)
            )
            call_kwargs["negative_prompt"] = negative_prompt or " "
        elif negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        if ref_bytes:
            images = [Image.open(BytesIO(b)).convert("RGB") for b in ref_bytes]
            # Qwen-Image-Edit family takes `image` (a list for the Plus pipeline).
            if spec.family == "qwen":
                call_kwargs["image"] = images if len(images) > 1 else images[0]
            else:
                call_kwargs["image"] = images[0]

        try:
            result = pipeline(**call_kwargs)
        except TypeError as exc:
            # A signature mismatch means our spec table is wrong for this
            # checkpoint. Say so plainly instead of masking it as a runtime error.
            raise ProviderError(
                f"{target} rejected the parameters we sent ({str(exc)[:200]}). "
                f"Its pipeline signature differs from MODEL_SPECS; update the spec.",
                provider=self.name,
                retryable=False,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            oom = "out of memory" in message.casefold() or "CUDA" in message
            raise ProviderError(
                f"{self.name} generation failed: {message[:300]}",
                provider=self.name,
                retryable=not oom,
            ) from exc

        images = getattr(result, "images", None) or []
        if not images:
            raise ProviderError(f"{self.name} produced no images.", provider=self.name)

        buffer = BytesIO()
        images[0].save(buffer, format="PNG")
        content = buffer.getvalue()

        return GeneratedImage(
            content=content,
            content_type=sniff_image_type(content),
            width=images[0].width,
            height=images[0].height,
            seed=seed,
            provider=self.name,
            model=target,
            cost=CostEstimate.zero(),
            metadata={
                "local": True,
                "family": spec.family,
                "steps": call_kwargs["num_inference_steps"],
                "guidance_scale": spec.guidance_scale,
                "true_cfg_scale": spec.true_cfg_scale,
                "identity_refs_used": len(ref_bytes),
                # Never store the reference bytes or any embedding (NFR-9).
            },
        )

    def _collect_refs(
        self, identity_refs: dict[str, list[bytes]] | None, structural_ref: bytes | None
    ) -> list[bytes]:
        """Gather reference bytes in a stable order, capped by the spec."""
        gathered: list[bytes] = []
        for person_key in sorted(identity_refs or {}):
            for content in (identity_refs or {})[person_key]:
                if content:
                    gathered.append(content)
        if structural_ref:
            gathered.append(structural_ref)
        limit = resolve_spec(self.model).identity_refs
        return gathered[:limit] if limit > 0 else gathered[:0]

    async def generate(self, **kwargs: Any) -> GeneratedImage:
        import asyncio  # noqa: PLC0415

        return await asyncio.to_thread(self.generate_sync, **kwargs)

    async def estimate_cost(self, *, count: int = 1, model: str | None = None) -> CostEstimate:
        return CostEstimate(credits=0, usd=0.0)

    def price_for(self, size: tuple[int, int], *, ref_count: int = 0) -> float:
        return 0.0

    def unload(self) -> None:
        """Free VRAM. Without this, a second Colab run often OOMs."""
        self.pipeline = None
        self._loaded_model = ""
        try:
            import torch  # noqa: PLC0415

            if self._has_cuda(torch):
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "DEFAULT_MODEL",
    "FP16_MODELS",
    "HuggingFaceImageProvider",
    "MODEL_SPECS",
    "ModelSpec",
    "QUICK_TEST_MODEL",
    "UNGATED_MODELS",
    "gated_help",
    "is_gated",
    "resolve_spec",
    "ungated_alternatives",
]
