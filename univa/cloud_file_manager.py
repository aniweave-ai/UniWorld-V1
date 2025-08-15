import io
import os
from pathlib import Path
from typing import Union

import boto3
from PIL import Image


class ImageType:
    ORIGINAL = "original"
    GPT = "gpt"
    GEMINI = "gemini"
    GPT_OPTIMIZED = "gpt_optimized"
    GEMINI_OPTIMIZED = "gemini_optimized"


class S3FileManager:
    def __init__(self, bucket_name: str, region_name: str):
        self.s3 = boto3.client("s3", region_name=region_name)
        self.bucket_name = bucket_name
        self.base_path = "clean-room/{version}/{task_id}/{image_type}/{name}"

    def get_s3_path(
        self, version: str, task_id: str, image_type: str, name: str
    ) -> str:
        return self.base_path.format(
            version=version, task_id=task_id, image_type=image_type, name=name
        )

    # === UPLOAD METHODS ===
    def upload_variant_image(
        self, image: Image.Image, version: str, task_id: str, image_type: str, name: str
    ):
        key = self.get_s3_path(version, task_id, image_type, name)
        self.upload_image(image, key)

    def upload_image(self, image: Image.Image, key: str, format: str = "JPEG"):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=format)
        image_bytes.seek(0)
        self.s3.upload_fileobj(image_bytes, self.bucket_name, key)

    def upload_file(self, file_path: Union[str, Path], key: str):
        self.s3.upload_file(str(file_path), self.bucket_name, key)

    def upload_folder(self, folder_path: Union[str, Path], s3_prefix: str = ""):
        """Upload all files in a folder to S3, preserving relative paths."""
        folder_path = Path(folder_path)
        for root, _, files in os.walk(folder_path):
            for file in files:
                local_path = Path(root) / file
                # Preserve the folder structure relative to folder_path
                relative_path = local_path.relative_to(folder_path)
                s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")
                self.upload_file(local_path, s3_key)
                print(f"Uploaded {local_path} → s3://{self.bucket_name}/{s3_key}")

    # === DOWNLOAD METHODS ===

    def download_file(self, local_path: Union[str, Path], key: str):
        self.s3.download_file(self.bucket_name, key, str(local_path))

    def download_image(self, key: str) -> Image.Image:
        """
        Downloads an image file from S3 directly into memory (PIL Image).
        """
        response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
        image_bytes = response["Body"].read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # === UTILITY METHODS ===
    def list_folders(self, prefix: str) -> list[str]:
        """
        Lists 'folders' (common prefixes) under a given S3 prefix.
        Example: prefix = 'clean-room/version123/task456/'
        Returns: ['clean-room/version123/task456/typeA/', 'clean-room/version123/task456/typeB/']
        """
        paginator = self.s3.get_paginator("list_objects_v2")
        result = []
        for page in paginator.paginate(
            Bucket=self.bucket_name, Prefix=prefix, Delimiter="/"
        ):
            if "CommonPrefixes" in page:
                result.extend(cp["Prefix"] for cp in page["CommonPrefixes"])
        return result

    def list_files(self, prefix: str) -> list[str]:
        """
        Lists all file keys under a given S3 prefix (non-recursively or recursively).
        Example: prefix = 'clean-room/version123/task456/typeA/'
        Returns: ['clean-room/version123/task456/typeA/img1.jpg', ...]
        """
        paginator = self.s3.get_paginator("list_objects_v2")
        result = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            if "Contents" in page:
                result.extend(obj["Key"] for obj in page["Contents"])
        return result


