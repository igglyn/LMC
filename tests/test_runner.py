from lmc.config import TeacherRuntimeConfig
from lmc.runner import GenerationRequest, GenerationResult, LocalTeacherRunner


class FakeBackend:
    def __init__(self):
        self.batches = []

    def generate_batch(self, requests):
        self.batches.append([request.id for request in requests])
        return [GenerationResult(request.id, request.prompt or "", 1, 1) for request in requests]


def test_runner_batches_logical_workers_against_one_backend():
    backend = FakeBackend()
    runner = LocalTeacherRunner(backend, TeacherRuntimeConfig(max_concurrent_requests=2, request_queue_size=2))
    results = runner.generate([GenerationRequest(id=str(index), prompt="test") for index in range(5)])
    assert backend.batches == [["0", "1"], ["2", "3"], ["4"]]
    assert [result.id for result in results] == ["0", "1", "2", "3", "4"]


def test_request_requires_one_prompt_form():
    try:
        GenerationRequest.from_dict({"id": "bad"})
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("invalid request was accepted")


def test_request_rejects_malformed_chat_message():
    try:
        GenerationRequest.from_dict({"id": "bad", "messages": [{"role": "user"}]})
    except ValueError as error:
        assert "role and content" in str(error)
    else:
        raise AssertionError("invalid message was accepted")
