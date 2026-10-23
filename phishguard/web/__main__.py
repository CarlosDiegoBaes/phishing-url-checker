"""``python -m phishguard.web`` serves the app on http://127.0.0.1:8000."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("phishguard.web:create_app", factory=True, host="127.0.0.1", port=8000)
