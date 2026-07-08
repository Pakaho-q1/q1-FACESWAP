from __future__ import annotations

import os
import shutil
from dataclasses import dataclass


PROJECT_FOLDER_NAME = "q1-FACESWAP"


def is_site_packages_path(path: str) -> bool:
    normalized = os.path.abspath(path).replace("/", "\\").lower()
    return "\\site-packages\\" in normalized or normalized.endswith("\\site-packages")


def normalize_project_root(project_path: str) -> str:
    raw = os.path.abspath(project_path)
    if os.path.basename(raw).lower() == PROJECT_FOLDER_NAME.lower():
        return raw
    return os.path.join(raw, PROJECT_FOLDER_NAME)


@dataclass(frozen=True)
class ProjectLayout:
    project_root: str
    assets_dir: str
    models_dir: str
    faces_dir: str
    temp_audio_dir: str
    tensorrt_home: str
    trt_cache_dir: str
    docs_dir: str
    env_path: str
    env_user_example_path: str
    env_user_path: str
    input_dir: str
    output_dir: str


def build_layout(project_root: str) -> ProjectLayout:
    assets = os.path.join(project_root, "assets")
    return ProjectLayout(
        project_root=project_root,
        assets_dir=assets,
        models_dir=os.path.join(assets, "models"),
        faces_dir=os.path.join(assets, "faces"),
        temp_audio_dir=os.path.join(assets, "temp_audio"),
        tensorrt_home=os.path.join(assets, "TensorRT"),
        trt_cache_dir=os.path.join(assets, "trt_cache"),
        docs_dir=os.path.join(assets, "docs"),
        env_path=os.path.join(assets, ".env"),
        env_user_example_path=os.path.join(assets, ".env_user.example"),
        env_user_path=os.path.join(assets, ".env_user"),
        input_dir=os.path.join(project_root, "input"),
        output_dir=os.path.join(project_root, "output"),
    )


def ensure_project_layout(project_root: str, source_assets_dir: str) -> ProjectLayout:
    layout = build_layout(project_root)

    for path in [
        layout.project_root,
        layout.assets_dir,
        layout.models_dir,
        layout.faces_dir,
        layout.temp_audio_dir,
        layout.tensorrt_home,
        layout.trt_cache_dir,
        layout.docs_dir,
        layout.input_dir,
        layout.output_dir,
    ]:
        os.makedirs(path, exist_ok=True)

    source_docs = os.path.join(source_assets_dir, "docs")
    if os.path.isdir(source_docs):
        for entry in os.listdir(source_docs):
            src = os.path.join(source_docs, entry)
            dst = os.path.join(layout.docs_dir, entry)
            if os.path.isdir(src):
                if not os.path.isdir(dst):
                    shutil.copytree(src, dst)
            elif os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy2(src, dst)

    for filename, target in [
        (".env", layout.env_path),
        (".env_user.example", layout.env_user_example_path),
        (".env_user", layout.env_user_path),
    ]:
        src = os.path.join(source_assets_dir, filename)
        if os.path.isfile(src) and not os.path.isfile(target):
            shutil.copy2(src, target)
        elif not os.path.exists(target):
            with open(target, "w", encoding="utf-8") as f:
                f.write("")

    return layout
