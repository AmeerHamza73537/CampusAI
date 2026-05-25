from app import app

def run_tests():
    c = app.test_client()
    samples = ["hi", "what courses do you offer", "college fees", "where is the hostel"]
    for msg in samples:
        r = c.post('/predict', json={'message': msg})
        print(msg, '->', r.status_code, r.get_json())

if __name__ == '__main__':
    run_tests()
