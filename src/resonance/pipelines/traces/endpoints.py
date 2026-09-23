"""Credential + endpoint resolution for teacher-trace generation.

Ported from run_dir('traces')/scripts/endpoints.py. The self-hosted Modal endpoint URL
is no longer hardcoded: set KIMI_MODAL_API_BASE (or pass --api-base).

Three routes, in descending order of how well they survive a 40,000-call run:

  modal       a self-hosted Modal-served K3. NO shared-tenant throttle, which is the only reason a
              two-pass 40k design is schedulable at all. Auth is a Modal WORKSPACE PROXY
              TOKEN: MODAL_KEY and MODAL_SECRET joined by a period. No single env var holds
              the joined value, so it is assembled here.
  native      api.moonshot.ai. Reads MOONSHOT_API_KEY, which does not exist on this box --
              KIMI_API_KEY does, so it is aliased.
  openrouter  measured too slow to finish: one run reached 790/5,000 in ~4 h sitting in
              32-second backoff (retry 6/8). Kept for fallback only, and MUST pin a provider
              or quantization varies per request and the corpus becomes heterogeneous.

Public endpoint URLs are defaults; the self-hosted one comes from the environment.
"""
import os

MODAL_BASE_KIMI = os.environ.get("KIMI_MODAL_API_BASE")  # https://<host>/v1
NATIVE_BASE_KIMI = "https://api.moonshot.ai/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MODEL_IDS = {
    "modal":      "moonshotai/Kimi-K3",
    "native":     "kimi-k3",
    "openrouter": "moonshotai/kimi-k3",
}

DEFAULT_BASES = {
    "modal":      MODAL_BASE_KIMI,
    "native":     NATIVE_BASE_KIMI,
    "openrouter": OPENROUTER_BASE,
}

NATIVE_KEY_ALIASES = ("MOONSHOT_API_KEY", "KIMI_API_KEY")
BASE_ENV_OVERRIDES = ("KIMI_API_BASE", "MOONSHOT_API_BASE")


class EndpointError(RuntimeError):
    pass


def resolve(route: str, api_base: str | None = None, model: str | None = None):
    """-> (model_id, api_base, api_key). Raises EndpointError with an actionable message."""
    if route not in DEFAULT_BASES:
        raise EndpointError(f"route must be one of {sorted(DEFAULT_BASES)}, got {route!r}")

    if route == "modal":
        mk, ms = os.environ.get("MODAL_KEY"), os.environ.get("MODAL_SECRET")
        if not (mk and ms):
            raise EndpointError(
                "route=modal needs MODAL_KEY and MODAL_SECRET in the environment. "
                "They are joined into a single 'wk-<id>.ws-<secret>' proxy token.")
        api_key = f"{mk}.{ms}"
    elif route == "native":
        api_key = next((os.environ[k] for k in NATIVE_KEY_ALIASES if os.environ.get(k)), None)
        if not api_key:
            raise EndpointError(f"route=native needs one of {NATIVE_KEY_ALIASES}.")
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EndpointError("route=openrouter needs OPENROUTER_API_KEY.")

    base = (api_base
            or next((os.environ[k] for k in BASE_ENV_OVERRIDES if os.environ.get(k)), None)
            or DEFAULT_BASES[route])
    if not base:
        raise EndpointError(f"route={route} has no base URL: set KIMI_MODAL_API_BASE or pass --api-base.")
    return (model or MODEL_IDS[route]), base, api_key


def describe(route: str, base: str, model: str) -> str:
    warn = ""
    if route == "openrouter":
        warn = ("  WARNING: openrouter throttled to 790/5,000 in ~4 h, and "
                "varies quantization per request unless a provider is pinned.")
    return f"route={route}  model={model}  base={base}{warn}"
