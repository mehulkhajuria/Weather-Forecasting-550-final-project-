from pathlib import Path
from weather550.cli import main


def test_cli_demo(tmp_path,capsys):
    main(["demo","--out",str(tmp_path/"smoke")])
    printed=capsys.readouterr().out
    assert "SYNTHETIC" in printed
    assert (tmp_path/"smoke"/"SYNTHETIC_numeric.npz").exists()
