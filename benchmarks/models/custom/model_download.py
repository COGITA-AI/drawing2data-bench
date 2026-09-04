from pathlib import Path
import gdown
import os

def download_model():
    file_id = "177lLcpWqapeBtIe2d33uXehCTQ8Vj1Rd"
    url = f"https://drive.google.com/uc?id={file_id}"
    path = str(Path(os.path.dirname(os.path.abspath(__file__))) / "detector.pth")

    gdown.download(url, quiet=False, output=path)