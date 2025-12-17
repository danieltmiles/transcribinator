import pytest
from pytest_mock import MockerFixture

from ai import process_audio


@pytest.mark.asyncio
async def test_process_audio(mocker: MockerFixture) -> None:
    async def mock_send_func(*_args, **_kwargs):
        pass
    progress_send_stream = mocker.MagicMock(new_callable=mocker.AsyncMock)
    progress_send_stream.send = mock_send_func
    transcript_send_stream = mocker.MagicMock(new_callable=mocker.AsyncMock)
    transcript_send_stream.send = mock_send_func
    await process_audio(
        #audio_file_path="test_data/example.m4a",
        audio_file_path="test_data/ruth_john_bill.m4a",
        min_segment_length=1.0,
        progress_send_stream=progress_send_stream,
        transcript_send_stream=transcript_send_stream,
    )