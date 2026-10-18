"""Write library versions and hardware info to results/environment.txt."""
import platform
from importlib.metadata import version

import torch

from common import RESULTS

PKGS = ["torch", "transformers", "datasets", "accelerate", "seqeval", "scikit-learn", "numpy", "pandas",
        "matplotlib", "seaborn", "pdfplumber", "python-docx", "fastapi", "uvicorn", "python-pptx"]

lines = [f"python {platform.python_version()}", f"platform {platform.platform()}",
         f"cpu {platform.processor()}",
         f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
         f"cuda {torch.version.cuda}", "seed 42", ""]
for p in PKGS:
    try:
        lines.append(f"{p}=={version(p)}")
    except Exception:
        lines.append(f"{p} (not installed)")
RESULTS.mkdir(exist_ok=True)
(RESULTS / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
