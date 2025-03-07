# run.py

from waitress import serve
from detect import app  # Import the app from detect.py

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8000)  # Change the host and port as needed
