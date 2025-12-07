"""Simple FastAPI server exposing an /analyze endpoint for the
echo-speech-module. This wraps the existing analysis functions so the
Android app (or any client) can upload audio and receive the analysis
results as JSON.

Usage:
  POST /analyze
    - multipart form-data
      - file: audio file (.wav)
      - intensity, speechrate, intonation, articulation: boolean flags (form fields)
      - ref_text: optional reference text for articulation
      - max_workers: optional int

The endpoint runs requested analyses concurrently and returns a JSON
object with keys for each requested module.
"""

import concurrent.futures
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from articulation import analyze_articulation
from intensity import analyze_intensity
from intonation import analyze_intonation
from response import ErrorResponse, Response
from speechrate import analyze_speechrate

app = FastAPI(title="Echo Speech Module API")


@app.get("/health")
def health():
    return {"status": "ok"}


def _to_dict(obj):
    """Convert Response or ErrorResponse to plain dict for JSON serialization."""
    try:
        if isinstance(obj, Response):
            return obj.get_data()
    except Exception:
        pass
    # If it's already a dict or JSONable, return as-is
    return obj


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    intensity: bool = Form(False),
    speechrate: bool = Form(False),
    intonation: bool = Form(False),
    articulation: bool = Form(False),
    ref_text: Optional[str] = Form(None),
    max_workers: int = Form(4),
):
    # Save uploaded file to a temporary path
    suffix = Path(file.filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp_path = Path(tmp.name)
    finally:
        tmp.close()

    tasks = {}
    if intensity:
        tasks["intensity"] = analyze_intensity
    if speechrate:
        tasks["speechrate"] = analyze_speechrate
    if intonation:
        tasks["intonation"] = analyze_intonation
    if articulation:
        # wrap articulation to include reference_text kwarg
        def _art_wrap(path: Path):
            return analyze_articulation(audio_file_path=path, reference_text=ref_text)

        tasks["articulation"] = _art_wrap

    if not tasks:
        os.unlink(tmp_path)
        raise HTTPException(
            status_code=400, detail="No analysis module selected")

    results = {}
    # run in threads to allow concurrency for I/O and network-bound STT
    max_workers = max(1, min(max_workers, len(tasks)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_name = {}
        for name, func in tasks.items():
            fut = ex.submit(func, tmp_path)
            fut_to_name[fut] = name

        for fut in concurrent.futures.as_completed(fut_to_name):
            name = fut_to_name[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = ErrorResponse(
                    error_name=e.__class__.__name__, error_details=str(e))
            results[name] = _to_dict(res)

    # cleanup temporary file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    return JSONResponse(content=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
