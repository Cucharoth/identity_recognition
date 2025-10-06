import pytest
from fastapi.testclient import TestClient

from app.app import create_app
from app.router import prediction_router
from app.services.validator_service import ValidatorService
from app.services.classifier_service import ClassifierService
from app.services import classifier_service as classifier_module


@pytest.fixture
def app_instance():
    app = create_app()
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(app_instance):
    with TestClient(app_instance) as test_client:
        yield test_client


def _override_services(app, validator=None, classifier=None):
    if validator is not None:
        app.dependency_overrides[
            prediction_router.get_validator_service
        ] = lambda: validator
    if classifier is not None:
        app.dependency_overrides[
            prediction_router.get_classifier_service
        ] = lambda: classifier


def _prepare_classifier(monkeypatch, score=0.93):
    class StubFaceModel:
        version = "test-model"

        def predict(self, image):
            self.last_image = image
            return score

    class StubModelStats:
        def __init__(self):
            self.records = []

        def record_inference(self, data):
            self.records.append(data)

    stub_face_model = StubFaceModel()
    stub_model_stats = StubModelStats()

    monkeypatch.setattr(classifier_module.FaceModel, "_instance", None, raising=False)
    monkeypatch.setattr(
        classifier_module.FaceModel,
        "get_instance",
        classmethod(lambda cls, *_, **__: stub_face_model),
        raising=False,
    )
    monkeypatch.setattr(classifier_module.ModelStats, "_instance", None, raising=False)
    monkeypatch.setattr(
        classifier_module.ModelStats,
        "get_instance",
        classmethod(lambda cls, *_, **__: stub_model_stats),
        raising=False,
    )
    monkeypatch.setattr(classifier_module.time, "sleep", lambda *_: None)

    return stub_face_model, stub_model_stats


def test_verify_success(app_instance, client, monkeypatch):
    _prepare_classifier(monkeypatch, score=0.93)

    validator = ValidatorService()
    classifier = ClassifierService()
    _override_services(app_instance, validator=validator, classifier=classifier)

    response = client.post(
        "/verify",
        files={"file": ("face.jpg", b"fake-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == classifier.model.version
    assert payload["data"]["is_me"] is True
    assert payload["metadata"]["request_id"]
    assert payload["metadata"]["timestamp"]


def test_verify_validation_error(app_instance, client, monkeypatch):
    _prepare_classifier(monkeypatch)

    validator = ValidatorService()
    classifier = ClassifierService()
    _override_services(app_instance, validator=validator, classifier=classifier)

    response = client.post(
        "/verify",
        files={"file": ("face.txt", b"fake-bytes", "text/plain")},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["detail"]["code"] == 400
    assert "Invalid file type" in payload["detail"]["error"]
    assert payload["detail"]["metadata"]["request_id"]
    assert payload["detail"]["metadata"]["timestamp"]


def test_verify_internal_server_error(app_instance, client, monkeypatch):
    _prepare_classifier(monkeypatch)

    validator = ValidatorService()
    classifier = ClassifierService()

    def boom(*_):
        raise RuntimeError("model kaboom")

    monkeypatch.setattr(classifier, "verify", boom)
    _override_services(app_instance, validator=validator, classifier=classifier)

    response = client.post(
        "/verify",
        files={"file": ("face.jpg", b"fake-bytes", "image/jpeg")},
    )

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"]["code"] == 500
    assert (
        payload["detail"]["error"]
        == "Lo sentimos, ha ocurrido un error procesando tu consulta."
    )
    assert payload["detail"]["metadata"]["request_id"]
    assert payload["detail"]["metadata"]["timestamp"]
