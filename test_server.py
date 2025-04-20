import requests
import sys

def test_server():
    """Test if the Django server is running."""
    try:
        # Try to connect to the test endpoint
        response = requests.get('http://127.0.0.1:8000/test-json/')
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return True
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is it running?")
        return False

if __name__ == "__main__":
    print("Testing Django server...")
    success = test_server()
    if not success:
        print("\nTo start the server, run:")
        print("python manage.py runserver")
        sys.exit(1)
    else:
        print("\nServer is running correctly!") 