import os
import pytest
from fastapi.testclient import TestClient
import db
from main import app

TEST_DB = "test_assistant.db"

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Points DB operations to a temporary test SQLite database and cleans it up after tests."""
    db.DB_FILE = TEST_DB
    db.init_db()
    yield
    # Teardown: delete the test database after all tests finish
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except PermissionError:
            pass

@pytest.fixture
def client():
    """FastAPI TestClient fixture for running API requests in-memory."""
    return TestClient(app)
