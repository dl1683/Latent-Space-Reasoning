import experiments.build_gated_attention_wsl_bootstrap as wsl_bootstrap
from experiments.build_gated_attention_wsl_bootstrap import build_wsl_bootstrap_plan, render_markdown


def test_wsl_bootstrap_blocks_missing_python_bootstrap_tools(monkeypatch):
    monkeypatch.setattr(wsl_bootstrap.shutil, "which", lambda name: "C:/Windows/System32/wsl.exe")
    monkeypatch.setattr(wsl_bootstrap, "_run", _fake_run_missing_bootstrap)

    plan = build_wsl_bootstrap_plan(distro="Ubuntu")
    markdown = render_markdown(plan)

    assert plan["ready_for_wsl_runtime_bootstrap"] is False
    assert "WSL Python has no pip" in plan["blocking_reasons"]
    assert "WSL Python lacks ensurepip/python3-venv" in plan["blocking_reasons"]
    assert "sudo requires a password" in plan["blocking_reasons"][2]
    assert "python3.12-venv" in "\n".join(plan["manual_install_commands"])
    assert "Ready for WSL runtime bootstrap: `False`" in markdown


def test_wsl_bootstrap_ready_when_gpu_python_pip_and_sudo_work(monkeypatch):
    monkeypatch.setattr(wsl_bootstrap.shutil, "which", lambda name: "C:/Windows/System32/wsl.exe")
    monkeypatch.setattr(wsl_bootstrap, "_run", _fake_run_ready)

    plan = build_wsl_bootstrap_plan(distro="Ubuntu")

    assert plan["ready_for_wsl_runtime_bootstrap"] is True
    assert plan["blocking_reasons"] == []
    assert "empty model ok" in plan["post_install_validation_command"]


def _fake_run_missing_bootstrap(cmd):
    joined = " ".join(cmd)
    if "-l -v" in joined:
        return _ok("Ubuntu Stopped 2")
    if "nvidia-smi" in joined:
        return _ok("NVIDIA GeForce RTX 5090 Laptop GPU, 24463, 0, 0")
    if "python3 --version" in joined:
        return _ok("/usr/bin/python3\nPython 3.12.3")
    if "python3 -m pip" in joined:
        return _fail("/usr/bin/python3: No module named pip")
    if "import ensurepip" in joined:
        return _fail("ModuleNotFoundError: No module named ensurepip")
    if "sudo -n true" in joined:
        return _fail("sudo: a password is required")
    return _ok("ok")


def _fake_run_ready(cmd):
    joined = " ".join(cmd)
    if "nvidia-smi" in joined:
        return _ok("NVIDIA GeForce RTX 5090 Laptop GPU, 24463, 0, 0")
    if "python3 --version" in joined:
        return _ok("/usr/bin/python3\nPython 3.12.3")
    if "python3 -m pip" in joined:
        return _ok("pip 24.0")
    if "import ensurepip" in joined:
        return _ok("ensurepip ok")
    if "sudo -n true" in joined:
        return _ok("")
    return _ok("ok")


def _ok(stdout):
    return {"returncode": 0, "stdout": stdout, "stderr": "", "command": ["fake"]}


def _fail(stderr):
    return {"returncode": 1, "stdout": "", "stderr": stderr, "command": ["fake"]}
