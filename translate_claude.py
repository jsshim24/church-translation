import sys
import time
import threading
from typing import Optional, Union

import anthropic

from main import OUTLINE_WRAPPER, build_prompt


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-6"
CACHE_MIN_TOKENS = 1024
KEEPALIVE_IDLE_SECONDS = 270  # 4m30s; stay under the 5-minute ephemeral TTL
KEEPALIVE_POLL_SECONDS = 10


# ── Client factory ────────────────────────────────────────────────────────────


def make_client(api_key: str) -> anthropic.Anthropic:
    """Construct the shared Anthropic client. Lets the orchestrator stay free
    of any direct `anthropic` symbol reference."""
    return anthropic.Anthropic(api_key=api_key)


# ── System prompt assembly ────────────────────────────────────────────────────


def build_system_blocks(base_prompt: str, outline: Optional[str],
                        cache: bool) -> Union[str, list[dict]]:
    """Assemble the system parameter for Anthropic.

    No outline  -> plain string (preserves existing API shape).
    With outline -> single TextBlockParam with optional cache_control.
    """
    if outline is None:
        return base_prompt
    combined = base_prompt + OUTLINE_WRAPPER.format(outline=outline)
    block: dict = {"type": "text", "text": combined}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


# ── Caching helpers ───────────────────────────────────────────────────────────


def count_system_tokens(client: anthropic.Anthropic,
                        system: Union[str, list[dict]],
                        model: str) -> int:
    """Exact token count for the system parameter by differencing against a
    baseline call with no system. Uses count_tokens (free of charge)."""
    dummy_msg = [{"role": "user", "content": "x"}]
    baseline = client.messages.count_tokens(model=model, messages=dummy_msg)
    full = client.messages.count_tokens(model=model, system=system, messages=dummy_msg)
    return full.input_tokens - baseline.input_tokens


def check_cache_eligibility(client: anthropic.Anthropic,
                            base_prompt: str, outline: str, model: str,
                            label: str = "") -> tuple[Union[str, list[dict]], bool]:
    """Return (system_blocks, cache_enabled). Warns and strips cache_control
    if the combined system tokens fall below CACHE_MIN_TOKENS. The optional
    `label` is included in log output to distinguish per-target workers."""
    candidate = build_system_blocks(base_prompt, outline, cache=True)
    tokens = count_system_tokens(client, candidate, model)
    tag = f" [{label}]" if label else ""
    if tokens >= CACHE_MIN_TOKENS:
        print(f"Prompt caching enabled{tag} ({tokens} system tokens).")
        return candidate, True
    print(
        f"Warning: system prompt + outline is {tokens} tokens{tag}, below the "
        f"{CACHE_MIN_TOKENS}-token caching threshold. Running without cache.",
        file=sys.stderr,
    )
    return build_system_blocks(base_prompt, outline, cache=False), False


def warm_cache(client: anthropic.Anthropic,
               system: Union[str, list[dict]], model: str,
               label: str = "") -> None:
    """One blocking call to populate the ephemeral cache. Exits on failure so
    an invalid API key or model name surfaces before the session starts."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1,
            system=system,
            messages=[{"role": "user", "content": "ready"}],
        )
    except Exception as e:
        sys.exit(f"Cache warmup failed: {e}")
    written = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    tag = f" [{label}]" if label else ""
    print(f"Cache warmed{tag} ({written} tokens written).")


# ── Backend ───────────────────────────────────────────────────────────────────


class Backend:
    """Claude translation backend.

    Owns the system prompt for one (source, target) pair, cache state,
    activity tracking, and (if cached) the keepalive thread. Pure
    translation API wrapper — has no knowledge of TranslationWorker's
    queue/[SKIP]/context shell.
    """

    def __init__(self, client: anthropic.Anthropic, source: str, target: str,
                 system: Union[str, list[dict]], cache_enabled: bool, model: str):
        self.client = client
        self.source = source
        self.target = target
        self.system = system
        self.cache_enabled = cache_enabled
        self.model = model
        self._last_activity = time.monotonic()
        self._activity_lock = threading.Lock()
        self._keepalive_thread: Optional[threading.Thread] = None

    @classmethod
    def from_outline(cls, client: anthropic.Anthropic, source: str, target: str,
                     outline: Optional[str], model: str) -> "Backend":
        prompt = build_prompt(source, target)
        if outline is None:
            system: Union[str, list[dict]] = prompt
            cache_enabled = False
        else:
            system, cache_enabled = check_cache_eligibility(
                client, prompt, outline, model, label=target
            )
        return cls(client=client, source=source, target=target,
                   system=system, cache_enabled=cache_enabled, model=model)

    def warmup(self) -> None:
        if self.cache_enabled:
            warm_cache(self.client, self.system, self.model, label=self.target)

    def mark_activity(self) -> None:
        with self._activity_lock:
            self._last_activity = time.monotonic()

    def _idle_seconds(self) -> float:
        with self._activity_lock:
            return time.monotonic() - self._last_activity

    def translate(self, context: list[tuple[str, str]], latest: str) -> str:
        messages: list[dict] = []
        for s, t in context:
            messages.append({"role": "user", "content": s})
            messages.append({"role": "assistant", "content": t})
        messages.append({"role": "user", "content": latest})
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system,
            messages=messages,
        )
        u = resp.usage
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        print(f"[usage {self.target}: in={u.input_tokens} cache_read={cr} "
              f"cache_write={cw} out={u.output_tokens}]", file=sys.stderr)
        out = resp.content[0].text.strip()
        self.mark_activity()
        return out

    def start_keepalive(self, stop_event: threading.Event) -> None:
        if not self.cache_enabled:
            return
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, args=(stop_event,), daemon=True
        )
        self._keepalive_thread.start()

    def _keepalive_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.wait(KEEPALIVE_POLL_SECONDS):
            try:
                if self._idle_seconds() < KEEPALIVE_IDLE_SECONDS:
                    continue
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1,
                    system=self.system,
                    messages=[{"role": "user", "content": "ready"}],
                )
                self.mark_activity()
                u = response.usage
                cr = getattr(u, "cache_read_input_tokens", 0) or 0
                cw = getattr(u, "cache_creation_input_tokens", 0) or 0
                print(f"[keepalive {self.target}: cache_read={cr} cache_write={cw}]",
                      file=sys.stderr)
            except Exception as e:
                print(f"[keepalive {self.target} error: {e}]", file=sys.stderr)
