import pytest

from yt_comments.cli.main import main


def test_cli_scrape_prints_video_id(capsys) -> None:
    exit_code = main(["scrape", "dQw4w9WgXcQ"]) # simulation of yt-comments scrape dQw4w9WgXcQ  
    assert exit_code == 0 # check the exit_code

    captured = capsys.readouterr()
    assert captured.out.strip() == "dQw4w9WgXcQ" # check that video id was printed


def test_cli_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0 # check that calling --help returns SystemExit(0)
    
def test_cli_requires_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2 # check that no command returns 2