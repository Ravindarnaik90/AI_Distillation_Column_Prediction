# -*- coding: utf-8 -*-
"""
run.py — convenience launcher.

Starts the FastAPI backend (which also serves the frontend) with a single command,
run from the project root:

    python run.py

This is equivalent to running, from the project root:

    uvicorn main:app --reload --host 127.0.0.1 --port 8000 --app-dir backend

(backend/main.py uses simple flat imports like `from model_service import ...`,
so uvicorn needs to be told that `backend/` is the import root via --app-dir /
app_dir — this script does that for you automatically.)
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir="backend")
