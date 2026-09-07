from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class BedrockClient:
    """Lazy boto3 Bedrock adapter; construction does not make a network call."""

    def __init__(
        self, profile_name: str = "aidev", region_name: str = "us-east-1"
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("Install boto3 to use Bedrock mode") from exc
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        self.client = session.client("bedrock-runtime")

    def converse(
        self, prompt: str, model_id: str, image_path: str | None = None
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"text": prompt}]
        if image_path:
            image_bytes = Path(image_path).read_bytes()
            content.append(
                {
                    "image": {
                        "format": Path(image_path).suffix.lstrip("."),
                        "source": {"bytes": image_bytes},
                    }
                }
            )
        response = self.client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"temperature": 0.0},
        )
        return response

    def embed(self, text: str, model_id: str) -> list[float]:
        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps({"inputText": text}),
            contentType="application/json",
            accept="application/json",
        )
        body = response["body"].read()
        return json.loads(body)["embedding"]

    @staticmethod
    def response_text(response: dict[str, Any]) -> str:
        return "".join(
            part.get("text", "")
            for part in response.get("output", {}).get("message", {}).get("content", [])
        )


def bedrock_config_from_env() -> dict[str, str]:
    return {
        "profile_name": os.getenv("AWS_PROFILE", "aidev"),
        "region_name": os.getenv("AWS_REGION", "us-east-1"),
        "text_model_id": os.getenv("RAMEA_TEXT_MODEL_ID", "openai.gpt-oss-120b-1:0"),
        "vision_model_id": os.getenv("RAMEA_VISION_MODEL_ID", "amazon.nova-lite-v1:0"),
        "embedding_model_id": os.getenv(
            "RAMEA_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
        ),
    }
