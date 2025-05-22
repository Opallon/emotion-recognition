from typing import Dict, Any
import json
from config import CLASS_MAPPING

class JsonHandler:
    @staticmethod
    def load(filename: str) -> Dict[str, Any]:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {k: 0 for k in CLASS_MAPPING.keys()}

    @staticmethod
    def save(filename: str, data: Dict[str, Any]) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
