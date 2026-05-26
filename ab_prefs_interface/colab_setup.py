"""Bootstrap ab_prefs_interface on Colab / Vertex Workbench with GCS-hosted GT + audio."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_BUCKET = "dd_tfx_full_transcripts"
DEFAULT_MOUNT = Path("/content/dd_tfx")
DEFAULT_REPO = Path("/content/ab_prefs_rating")
DEFAULT_REPO_URL = "https://github.com/rosyvs/ab_prefs_rating.git"

ASR_PROVIDER_SUBDIRS = {
    "aai_up3": "aai/universal-3-pro",
    "aai_up3_v2": "aai/universal-3-pro-v2",
    "aai_up3_extraprompted": "aai/universal-3-pro_extraprompted",
    "azure_fast": "azure/fast-transcription",
    "azure_llm": "azure/llm-transcribe",
    "azure_llm_extraprompted": "azure/llm-transcribe_extraprompted",
    "azure_mai": "azure/mai-transcribe-1",
    "deepgram_nova3": "deepgram/nova-3",
    "gemini_31_pro_extraprompted": "gemini/gemini-3.1-pro-preview_extraprompted",
}


def in_colab() -> bool:
    return "google.colab" in sys.modules


def colab_auth() -> None:
    if not in_colab():
        return
    from google.colab import auth  # type: ignore

    auth.authenticate_user()
    print("Google auth OK — gcsfuse/gsutil will use your credentials (not public bucket access)")


def verify_gcs_access(bucket: str) -> None:
    """Fail unless current credentials can read the bucket (blocks anonymous/public-only paths)."""
    probe = f"gs://{bucket}/transcripts/"
    result = subprocess.run(["gsutil", "ls", probe], capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise PermissionError(
            f"No authenticated access to {probe}. "
            f"Sign in with a Google account that has storage.objectViewer on gs://{bucket}. "
            f"Do not make this bucket public.\n{err}"
        )
    print(f"Verified IAM access to gs://{bucket}")


def install_gcsfuse() -> None:
    if shutil.which("gcsfuse"):
        return
    keyring = Path("/usr/share/keyrings/cloud.google.asc")
    if not keyring.is_file():
        subprocess.run(
            ["curl", "-fsSL", "https://packages.cloud.google.com/apt/doc/apt-key.gpg", "-o", str(keyring)],
            check=True,
        )
    codename = subprocess.run(["lsb_release", "-c", "-s"], capture_output=True, text=True, check=True).stdout.strip()
    repos = [f"gcsfuse-{codename}", "gcsfuse-jammy", "gcsfuse-bookworm", "gcsfuse-main"]
    last_err: subprocess.CalledProcessError | None = None
    for repo_name in repos:
        Path("/etc/apt/sources.list.d/gcsfuse.list").write_text(
            f"deb [signed-by={keyring}] https://packages.cloud.google.com/apt {repo_name} main\n",
            encoding="utf-8",
        )
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        try:
            subprocess.run(["apt-get", "install", "-qq", "-y", "fuse", "gcsfuse"], check=True)
            print(f"Installed gcsfuse from apt repo {repo_name}")
            return
        except subprocess.CalledProcessError as err:
            last_err = err
    raise RuntimeError("Could not install gcsfuse from Google apt repos") from last_err


def gcsfuse_mount(bucket: str, mount: Path) -> None:
    if not shutil.which("gcsfuse"):
        raise RuntimeError("gcsfuse not installed")
    mount.mkdir(parents=True, exist_ok=True)
    cmd = ["gcsfuse", "--implicit-dirs", bucket, str(mount)]
    print("Mounting:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not (mount / "transcripts").is_dir() or not (mount / "audio").is_dir():
        raise FileNotFoundError(f"Expected {mount}/transcripts and {mount}/audio after gcsfuse mount")


def workbench_bucket_path(bucket: str, mount_point: Path) -> Path | None:
    """Vertex Workbench: bucket already mounted via VM service account."""
    for candidate in (
        mount_point,
        Path(f"/home/jupyter/gcs/{bucket}"),
        Path(f"/gcs/{bucket}"),
    ):
        if (candidate / "transcripts").is_dir() and (candidate / "audio").is_dir():
            print(f"Using Workbench bucket path: {candidate}")
            return candidate
    return None


def mount_gcs_bucket(
    bucket: str = DEFAULT_BUCKET,
    mount_point: Path | str = DEFAULT_MOUNT,
) -> Path:
    """Mount bucket with gcsfuse after Google auth + IAM verify. No gsutil copy fallback."""
    mount = Path(mount_point)
    wb = workbench_bucket_path(bucket, mount)
    if wb is not None:
        return wb
    colab_auth()
    verify_gcs_access(bucket)
    install_gcsfuse()
    gcsfuse_mount(bucket, mount)
    return mount


def install_runtime_deps(repo_root: Path | str | None = None) -> None:
    if repo_root is not None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(Path(repo_root).resolve())],
            check=True,
        )
    if in_colab():
        subprocess.run(["apt-get", "install", "-qq", "-y", "ffmpeg"], check=True)
        from google.colab import output  # type: ignore

        output.enable_custom_widget_manager()


def clone_repo(
    dest: Path | str = DEFAULT_REPO,
    repo_url: str = DEFAULT_REPO_URL,
    branch: str = "main",
) -> Path:
    dest = Path(dest)
    if (dest / "pyproject.toml").is_file() and (dest / "ab_prefs_interface").is_dir():
        print(f"Repo already at {dest}")
        return dest
    if dest.exists():
        raise FileExistsError(f"{dest} exists but is not ab_prefs_rating — remove or pick another path")
    subprocess.run(["git", "clone", "--branch", branch, "--depth", "1", repo_url, str(dest)], check=True)
    return dest


def repo_paths(
    mount_root: Path | str,
    repo_root: Path | str,
    *,
    asr_subdir: str = "asr/dd210",
) -> dict[str, Path]:
    mount = Path(mount_root)
    repo = Path(repo_root)
    asr = mount / asr_subdir
    out = repo / "results" / "ab_prefs"
    return {
        "notebook_root": repo,
        "gt_dir": mount / "transcripts",
        "audio_dir": mount / "audio",
        "asr_root": asr,
        "config_json": repo / "configs" / "ab_prefs.providers.colab.json",
        "session_manifest": repo / "configs" / "ab_prefs.manifest.json",
        "clip_dir": out / "audio_clips",
        "cache_dir": out / "unit_cache",
        "output_dir": out,
    }


def write_colab_session_config(
    paths: dict[str, Path],
    dest: Path | str | None = None,
    *,
    compare_providers: str = "aai_up3_v2,azure_fast,deepgram_nova3",
    session_items: int = 30,
    demo_recordings: int = 5,
) -> Path:
    """Write runtime session JSON with GCS-backed paths."""
    dest = Path(dest or paths["notebook_root"] / "configs" / "ab_prefs.session.runtime.json")
    payload = {
        "notebook_root": str(paths["notebook_root"]),
        "gt_dir": str(paths["gt_dir"]),
        "audio_dir": str(paths["audio_dir"]),
        "config_json": str(paths["config_json"]),
        "session_manifest": str(paths["session_manifest"]),
        "clip_dir": str(paths["clip_dir"]),
        "cache_dir": str(paths["cache_dir"]),
        "output_dir": str(paths["output_dir"]),
        "ground_truth_name": "ground_truth",
        "include_ground_truth": True,
        "strategy": "random",
        "session_items": session_items,
        "seed": 7,
        "demo_recordings": demo_recordings,
        "demo_seed": 7,
        "min_gt_words": 5,
        "min_audio_seconds": 3.0,
        "compare_providers": compare_providers,
        "show_note": False,
        "show_providers": False,
        "verbose": True,
        "rebuild_cache": False,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def patch_providers_colab(asr_root: Path | str, providers_path: Path | str) -> None:
    """Rewrite providers.colab.json ASR dirs to ``{asr_root}/...`` (idempotent per mount)."""
    asr_root = Path(asr_root)
    path = Path(providers_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["providers"] = {name: str(asr_root / sub) for name, sub in ASR_PROVIDER_SUBDIRS.items()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def bootstrap(
    rater_id: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    mount_point: Path | str = DEFAULT_MOUNT,
    repo_root: Path | str = DEFAULT_REPO,
    repo_url: str = DEFAULT_REPO_URL,
    asr_subdir: str = "asr/dd210",
) -> tuple[Path, Path]:
    """Clone repo, install deps, auth + gcsfuse mount, write runtime session config."""
    repo = clone_repo(repo_root, repo_url=repo_url)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    install_runtime_deps(repo)
    mount = mount_gcs_bucket(bucket, mount_point)
    paths = repo_paths(mount, repo, asr_subdir=asr_subdir)
    patch_providers_colab(paths["asr_root"], paths["config_json"])
    session_path = write_colab_session_config(paths)
    print(f"GT: {paths['gt_dir']}")
    print(f"Audio: {paths['audio_dir']}")
    print(f"ASR: {paths['asr_root']}")
    print(f"Preferences → {paths['output_dir'] / f'preferences_{rater_id.strip()}.json'}")
    return repo, session_path
