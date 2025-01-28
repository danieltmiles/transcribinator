import pytest
from pytest_mock import MockerFixture

from ai import process_audio


@pytest.mark.asyncio
async def test_process_audio(mocker: MockerFixture) -> None:
    await process_audio(
        audio_file_path="test_data/example.m4a",
        num_speakers=3,
        min_segment_length=1.0,
        progress_send_stream=mocker.MagicMock(new_callable=mocker.AsyncMock),
        transcript_send_stream=mocker.MagicMock(new_callable=mocker.AsyncMock),
    )