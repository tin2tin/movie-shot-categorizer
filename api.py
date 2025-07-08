from fastapi import FastAPI, File, UploadFile
from shot_categorizer.categorizer import load_model, categorize_shot
import shutil
import os

app = FastAPI()
model, processor = load_model()

@app.post("/categorize-shot/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the shot categorization.
    """
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = categorize_shot(temp_file_path, model, processor)
    os.remove(temp_file_path)
    return results

@app.get("/")
def read_root():
    return {"message": "Welcome to the Shot Categorizer API"}