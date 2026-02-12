import pytest

from yt_comments.cli.main import main


def test_cli_scrape_prints_video_id(capsys) -> None:
    video_id = "dQw4w9WgXcQ" 
    
    exit_code = main(["scrape", "dQw4w9WgXcQ"]) # simulation of yt-comments scrape dQw4w9WgXcQ  
    assert exit_code == 0 # check the exit_code

    captured = capsys.readouterr()
    output = captured.out.strip()
    assert "Saved" in output # check that the json was saved
    assert video_id in output # check that video id was printed
    assert ".jsonl" in output # check that the file was created


def test_cli_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0 # check that calling --help returns SystemExit(0)
    
def test_cli_requires_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2 # check that no command returns 2