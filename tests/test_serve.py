from src.serve import COMMANDS, command


def test_default_is_api():
    assert command([]) == COMMANDS["api"]


def test_named_commands():
    assert command(["demo"])[:3] == ["streamlit", "run", "app.py"]


def test_other_commands_pass_through():
    assert command(["pytest", "-q"]) == ["pytest", "-q"]
