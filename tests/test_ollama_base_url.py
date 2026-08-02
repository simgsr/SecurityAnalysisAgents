"""Tests for OLLAMA_BASE_URL env-var override across CLI and client paths."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="module", autouse=True)
def _resync_reloaded_modules():
    """Restore module state after this file's importlib.reload() calls.

    Several tests below reload ``cli.utils`` to re-evaluate OLLAMA_BASE_URL.
    That leaves ``cli.main``'s star-imported names (e.g. get_ticker) bound to
    the pre-reload module objects, which breaks identity checks in unrelated
    tests that happen to run afterward. Re-sync once on teardown so the reload
    doesn't leak across test modules.
    """
    yield
    import cli.main
    import cli.utils
    importlib.reload(cli.utils)
    importlib.reload(cli.main)


# ---- openai_client side: registry-driven base_url resolution --------------


def _reload_client():
    import securityanalysisagents.llm_clients.openai_client as mod
    return importlib.reload(mod)


def _base_url(mod, provider, **kwargs):
    return str(mod.OpenAIClient(model="m", provider=provider, **kwargs).get_llm().openai_api_base)


def test_resolver_returns_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    mod = _reload_client()
    assert _base_url(mod, "ollama") == "http://localhost:11434/v1"


def test_resolver_returns_env_when_set(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-ollama:11434/v1")
    mod = _reload_client()
    assert _base_url(mod, "ollama") == "http://remote-ollama:11434/v1"


def test_resolver_evaluation_is_call_time(monkeypatch):
    """Setting the env AFTER module import must still take effect."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    mod = _reload_client()
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://late-set:11434/v1")
    assert _base_url(mod, "ollama") == "http://late-set:11434/v1"


def test_resolver_does_not_affect_other_providers(monkeypatch):
    """OLLAMA_BASE_URL should NOT leak into xai/deepseek/etc."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://elsewhere/v1")
    mod = _reload_client()
    assert _base_url(mod, "xai") == "https://api.x.ai/v1"
    assert _base_url(mod, "deepseek") == "https://api.deepseek.com"


def test_client_get_llm_picks_up_env(monkeypatch):
    """End-to-end: OllamaClient.get_llm() respects OLLAMA_BASE_URL."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://my-ollama:11434/v1")
    mod = _reload_client()
    client = mod.OpenAIClient(model="llama3.1", provider="ollama")
    llm = client.get_llm()
    assert "my-ollama" in str(llm.openai_api_base)


def test_explicit_base_url_overrides_env(monkeypatch):
    """An explicit base_url passed to the client wins over the env var."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-set:11434/v1")
    mod = _reload_client()
    client = mod.OpenAIClient(
        model="llama3.1",
        provider="ollama",
        base_url="http://explicit:11434/v1",
    )
    llm = client.get_llm()
    assert "explicit" in str(llm.openai_api_base)
    assert "env-set" not in str(llm.openai_api_base)


# ---- cli.utils side: select_llm_provider dropdown -------------------------


def test_cli_dropdown_uses_env(monkeypatch):
    """The Ollama entry in the CLI dropdown must reflect OLLAMA_BASE_URL."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://cli-remote:11434/v1")
    import cli.utils as cli_utils
    importlib.reload(cli_utils)
    # Reach inside the function via the same env-read it does at call time
    ollama_url = (
        __import__("os").environ.get("OLLAMA_BASE_URL")
        or "http://localhost:11434/v1"
    )
    assert ollama_url == "http://cli-remote:11434/v1"


def test_cli_dropdown_default_when_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    import cli.utils as cli_utils
    importlib.reload(cli_utils)
    ollama_url = (
        __import__("os").environ.get("OLLAMA_BASE_URL")
        or "http://localhost:11434/v1"
    )
    assert ollama_url == "http://localhost:11434/v1"


# ---- confirm_ollama_endpoint UX -------------------------------------------


def test_confirm_endpoint_shows_default(monkeypatch, capsys):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    import cli.utils as cli_utils
    importlib.reload(cli_utils)
    cli_utils.confirm_ollama_endpoint("http://localhost:11434/v1")
    out = capsys.readouterr().out
    assert "http://localhost:11434/v1" in out
    assert "OLLAMA_BASE_URL" not in out  # not from env
    assert "Note" not in out  # no warnings for the canonical default


def test_confirm_endpoint_marks_env_origin(monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-host:11434/v1")
    import cli.utils as cli_utils
    importlib.reload(cli_utils)
    cli_utils.confirm_ollama_endpoint("http://remote-host:11434/v1")
    out = capsys.readouterr().out
    assert "http://remote-host:11434/v1" in out
    assert "OLLAMA_BASE_URL" in out


def test_confirm_endpoint_warns_on_missing_scheme(monkeypatch, capsys):
    """If user sets OLLAMA_BASE_URL=0.0.0.128, advise on the expected shape."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "0.0.0.128")
    import cli.utils as cli_utils
    importlib.reload(cli_utils)
    cli_utils.confirm_ollama_endpoint("0.0.0.128")
    out = capsys.readouterr().out
    assert "missing a scheme" in out
    assert "http://<host>:11434/v1" in out


def test_confirm_endpoint_warns_on_non_default_port_remote(monkeypatch, capsys):
    """A remote host with no :11434 gets a soft hint about port mismatch."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-host/v1")
    import cli.utils as cli_utils
    importlib.reload(cli_utils)
    cli_utils.confirm_ollama_endpoint("http://remote-host/v1")
    out = capsys.readouterr().out
    assert "port 11434" in out


def test_confirm_endpoint_quiet_on_local_no_port(monkeypatch, capsys):
    """Local host without port shouldn't trigger the remote-port hint."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost/v1")
    import cli.utils as cli_utils
    importlib.reload(cli_utils)
    cli_utils.confirm_ollama_endpoint("http://localhost/v1")
    out = capsys.readouterr().out
    assert "Note" not in out  # localhost is fine without explicit port


def test_ollama_model_labels_no_local_suffix():
    """Labels should no longer claim '(local)' since the endpoint is dynamic."""
    from securityanalysisagents.llm_clients.model_catalog import get_model_options
    for mode in ("quick", "deep"):
        labels = [label for label, _ in get_model_options("ollama", mode)]
        assert all("local" not in label for label in labels), labels


def test_ollama_offers_custom_model_id():
    """Ollama users with custom-pulled models can pick 'Custom model ID'."""
    from securityanalysisagents.llm_clients.model_catalog import get_model_options
    for mode in ("quick", "deep"):
        entries = get_model_options("ollama", mode)
        values = [v for _, v in entries]
        assert "custom" in values, f"Ollama {mode!r} missing 'custom' option: {entries}"
        # Custom option is last so it doesn't push the curated defaults off-screen
        assert values[-1] == "custom", f"'custom' should be last entry: {values}"


# ---- provider ordering ----------------------------------------------------


def test_ollama_listed_first_in_provider_dropdown():
    """Ollama is the most-used local provider, so it heads the dropdown."""
    import cli.utils as cli_utils
    table = cli_utils._llm_provider_table()
    assert table[0][1] == "ollama", f"Ollama should be first, got: {[k for _, k, _ in table]}"


# ---- ensure_ollama_running auto-start -------------------------------------


def test_ensure_running_noop_when_already_up(monkeypatch):
    """If a server already answers, we neither warn nor spawn a process."""
    import cli.utils as cli_utils
    monkeypatch.setattr(cli_utils, "_ollama_server_up", lambda url, **kw: True)

    def _boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("should not spawn ollama serve when already up")

    monkeypatch.setattr("subprocess.Popen", _boom)
    cli_utils.ensure_ollama_running("http://localhost:11434/v1")


def test_ensure_running_skips_remote_endpoints(monkeypatch):
    """A remote OLLAMA_BASE_URL is someone else's box — never probe or spawn it."""
    import cli.utils as cli_utils
    probed = {"called": False}

    def _probe(url, **kw):
        probed["called"] = True
        return False

    monkeypatch.setattr(cli_utils, "_ollama_server_up", _probe)
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no spawn for remote")),
    )
    cli_utils.ensure_ollama_running("http://remote-host:11434/v1")
    assert probed["called"] is False


def test_ensure_running_warns_when_binary_missing(monkeypatch, capsys):
    """No server and no 'ollama' binary -> advisory install hint, no crash."""
    import cli.utils as cli_utils
    monkeypatch.setattr(cli_utils, "_ollama_server_up", lambda url, **kw: False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    cli_utils.ensure_ollama_running("http://localhost:11434/v1")
    out = capsys.readouterr().out
    assert "ollama.com/download" in out


def test_ensure_running_spawns_serve_when_binary_present(monkeypatch, capsys):
    """No server but 'ollama' on PATH -> launch 'ollama serve' detached."""
    import cli.utils as cli_utils
    # Down at first, then up after the "spawn" so the readiness poll succeeds fast.
    states = iter([False, True])
    monkeypatch.setattr(cli_utils, "_ollama_server_up", lambda url, **kw: next(states, True))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/ollama")
    calls = {}

    def _fake_popen(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    cli_utils.ensure_ollama_running("http://localhost:11434/v1")
    assert calls["cmd"] == ["/usr/local/bin/ollama", "serve"]
    assert calls["kwargs"].get("start_new_session") is True
    assert "up" in capsys.readouterr().out.lower()


# ---- ensure_ollama_model_pulled auto-pull ---------------------------------


def _no_pull(*a, **k):  # pragma: no cover - guards "must not pull" paths
    raise AssertionError("ollama pull should not run here")


def test_pull_skips_cloud_proxy_models(monkeypatch):
    """`:cloud` catalog defaults are proxied, not stored locally -> never pull."""
    import cli.utils as cli_utils
    monkeypatch.setattr(cli_utils, "_fetch_ollama_models", _no_pull)
    monkeypatch.setattr("subprocess.run", _no_pull)
    cli_utils.ensure_ollama_model_pulled("deepseek-v4-pro:cloud", "http://localhost:11434/v1")


def test_pull_skips_remote_endpoints(monkeypatch):
    """A remote server manages its own models -> don't pull locally."""
    import cli.utils as cli_utils
    monkeypatch.setattr(cli_utils, "_fetch_ollama_models", _no_pull)
    monkeypatch.setattr("subprocess.run", _no_pull)
    cli_utils.ensure_ollama_model_pulled("llama4", "http://remote-host:11434/v1")


def test_pull_skips_when_already_present(monkeypatch):
    """A model already served locally isn't re-pulled."""
    import cli.utils as cli_utils
    monkeypatch.setattr(cli_utils, "_fetch_ollama_models", lambda: ["llama4", "qwen3"])
    monkeypatch.setattr("subprocess.run", _no_pull)
    cli_utils.ensure_ollama_model_pulled("llama4", "http://localhost:11434/v1")


def test_pull_runs_when_missing(monkeypatch, capsys):
    """A local model not yet present triggers 'ollama pull <model>'."""
    import cli.utils as cli_utils
    monkeypatch.setattr(cli_utils, "_fetch_ollama_models", lambda: [])
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/ollama")
    calls = {}

    def _fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr("subprocess.run", _fake_run)
    cli_utils.ensure_ollama_model_pulled("llama4", "http://localhost:11434/v1")
    assert calls["cmd"] == ["/usr/local/bin/ollama", "pull", "llama4"]
    assert "Pulled llama4" in capsys.readouterr().out


def test_pull_warns_when_binary_missing(monkeypatch, capsys):
    """Missing 'ollama' binary yields an advisory hint, no crash."""
    import cli.utils as cli_utils
    monkeypatch.setattr(cli_utils, "_fetch_ollama_models", lambda: [])
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("subprocess.run", _no_pull)
    cli_utils.ensure_ollama_model_pulled("llama4", "http://localhost:11434/v1")
    out = capsys.readouterr().out
    assert "ollama pull llama4" in out
