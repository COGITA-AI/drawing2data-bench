from pathlib import Path
import gdown
import os

def download_model():
    file_id = "1xlMXgoJdesRE-ZGWBdMTXVo4GyD0JEFC"
    url = f"https://drive.google.com/uc?id={file_id}"
    path = str(Path(os.path.dirname(os.path.abspath(__file__))) / "detector.pth")

    gdown.download(url, quiet=False, output=path)