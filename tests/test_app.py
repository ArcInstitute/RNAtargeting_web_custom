import unittest
from streamlit.testing.v1 import AppTest

class TestApp(unittest.TestCase):
    def test_app(self):
        at = AppTest.from_file('../app.py')
        at.run(timeout=10)
        assert not at.exception

if __name__ == '__main__':
    unittest.main()