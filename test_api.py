from app import app


def run_tests():
    c = app.test_client()
    samples = ["hi", "what courses do you offer", "college fees", "where is the hostel"]
    for msg in samples:
        response = c.post("/predict", json={"message": msg})
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["response"]
        assert 0 <= payload["confidence"] <= 1
        print(msg, "->", response.status_code, payload)

    empty = c.post("/predict", json={"message": " "})
    assert empty.status_code == 400

    health = c.get("/health")
    assert health.status_code == 200
    assert health.get_json() == {"status": "ok"}


if __name__ == "__main__":
    run_tests()
