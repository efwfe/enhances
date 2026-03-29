"""
ComfyUI HTTP API client.

Provides image upload, workflow queuing, result polling, and image download.
No third-party dependencies beyond stdlib + opencv/numpy (already required by
the rest of this package).
"""

from __future__ import annotations

import json
import time
import uuid
import urllib.request
import urllib.parse
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ComfyUIClient:
    """
    Minimal synchronous HTTP client for a running ComfyUI backend.

    Args:
        host (str): ComfyUI hostname or IP.
        port (int): ComfyUI port (default 8188).
        poll_interval (float): Seconds between history-poll requests.
        timeout (float): Max seconds to wait for a prompt to complete.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        poll_interval: float = 0.5,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.client_id = str(uuid.uuid4())
        self.poll_interval = poll_interval
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Low-level HTTP helpers (stdlib only)
    # ------------------------------------------------------------------

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())

    def _post_json(self, path: str, payload: dict) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def _post_multipart(
        self,
        path: str,
        fields: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
    ) -> Any:
        """
        Send a multipart/form-data POST request.

        files: {field_name: (filename, bytes_data, content_type)}
        """
        boundary = uuid.uuid4().hex
        body = io.BytesIO()

        for name, value in fields.items():
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.write(f"{value}\r\n".encode())

        for name, (filename, data, content_type) in files.items():
            body.write(f"--{boundary}\r\n".encode())
            body.write(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            )
            body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
            body.write(data)
            body.write(b"\r\n")

        body.write(f"--{boundary}--\r\n".encode())
        body_bytes = body.getvalue()

        url = f"{self.base_url}{path}"
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_image(self, image: np.ndarray, name: str = "input.png") -> str:
        """
        Upload a numpy RGB image to the ComfyUI input directory.

        Args:
            image: HxWx3 uint8 RGB array.
            name:  Filename to use on the server (used as reference in workflows).

        Returns:
            The server-side filename (may differ from *name* if de-duped).
        """
        success, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Failed to encode image for upload")
        result = self._post_multipart(
            "/upload/image",
            fields={"overwrite": "true"},
            files={"image": (name, buf.tobytes(), "image/png")},
        )
        return result["name"]

    def queue_prompt(self, workflow: dict) -> str:
        """
        Submit a workflow dict to the ComfyUI prompt queue.

        Returns:
            The prompt_id string assigned by ComfyUI.
        """
        payload = {"prompt": workflow, "client_id": self.client_id}
        result = self._post_json("/prompt", payload)
        if "error" in result:
            raise RuntimeError(f"ComfyUI rejected workflow: {result['error']}")
        return result["prompt_id"]

    def wait_for_result(self, prompt_id: str) -> dict:
        """
        Poll /history until the prompt finishes.

        Returns:
            The history entry dict for this prompt_id.

        Raises:
            TimeoutError: if the prompt does not complete within self.timeout.
        """
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            history = self._get(f"/history/{prompt_id}")
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"ComfyUI prompt {prompt_id!r} did not complete within {self.timeout}s"
        )

    def download_image(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> np.ndarray:
        """
        Download a ComfyUI output image by filename.

        Returns:
            HxWx3 uint8 RGB numpy array.
        """
        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        url = f"{self.base_url}/view?{params}"
        with urllib.request.urlopen(url) as resp:
            raw = np.frombuffer(resp.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Could not decode downloaded image: {filename!r}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def run_workflow(self, workflow: dict) -> list[np.ndarray]:
        """
        Queue *workflow*, wait for completion, and return all output images.

        Returns:
            List of HxWx3 uint8 RGB arrays (one per output image node).
        """
        prompt_id = self.queue_prompt(workflow)
        result = self.wait_for_result(prompt_id)
        images: list[np.ndarray] = []
        for node_output in result.get("outputs", {}).values():
            for img_info in node_output.get("images", []):
                img = self.download_image(
                    img_info["filename"],
                    img_info.get("subfolder", ""),
                    img_info.get("type", "output"),
                )
                images.append(img)
        return images
