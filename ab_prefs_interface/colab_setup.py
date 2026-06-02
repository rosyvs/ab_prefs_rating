"""Bootstrap ab_prefs_interface on Colab / Vertex Workbench with GCS-hosted GT + audio."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ab_prefs_interface.session_config import parse_compare_providers

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
    return bool(os.environ.get("COLAB_RELEASE_TAG"))


def use_colab_html_ui() -> bool:
    """Colab pre-imports ipywidgets before user cells; HTML+callbacks work reliably."""
    return in_colab()


def colab_auth() -> None:
    try:
        from google.colab import auth  # type: ignore
    except ImportError:
        return
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
            f"Sign in with a Google account that has storage.objectViewer on gs://{bucket}.\n{err}"
        )
    print(f"Verified IAM access to gs://{bucket}")


def install_gcsfuse() -> None:
    if shutil.which("gcsfuse"):
        return
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-qq", "-y", "curl", "fuse", "lsb-release"], check=True)
    keyring = Path("/usr/share/keyrings/cloud.google.asc")
    if not keyring.is_file():
        subprocess.run(
            ["curl", "-fsSL", "https://packages.cloud.google.com/apt/doc/apt-key.gpg", "-o", str(keyring)],
            check=True,
        )
    codename = subprocess.run(["lsb_release", "-c", "-s"], capture_output=True, text=True, check=True).stdout.strip()
    repos = [f"gcsfuse-{codename}", "gcsfuse-jammy", "gcsfuse-bookworm", "gcsfuse-main"]
    for repo_name in repos:
        Path("/etc/apt/sources.list.d/gcsfuse.list").write_text(
            f"deb [signed-by={keyring}] https://packages.cloud.google.com/apt {repo_name} main\n",
            encoding="utf-8",
        )
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        if subprocess.run(["apt-get", "install", "-qq", "-y", "gcsfuse"], check=False).returncode == 0:
            print(f"Installed gcsfuse from apt repo {repo_name}")
            return
    # Colab default apt often has no gcsfuse package — install .deb from GitHub releases
    version = "3.9.1"
    deb = Path(f"/tmp/gcsfuse_{version}_amd64.deb")
    url = f"https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v{version}/gcsfuse_{version}_amd64.deb"
    print(f"apt install failed; downloading {url}")
    subprocess.run(["curl", "-fsSL", "-o", str(deb), url], check=True)
    subprocess.run(["dpkg", "-i", str(deb)], check=True)
    if not shutil.which("gcsfuse"):
        raise RuntimeError("gcsfuse install failed (apt + .deb fallback)")
    print(f"Installed gcsfuse {version} from GitHub release")


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


def enable_colab_widgets() -> None:
    """Enable ipywidgets on Jupyter/Workbench (not used on Colab — see use_colab_html_ui)."""
    if use_colab_html_ui() or not in_colab():
        return
    from google.colab import output  # type: ignore

    output.enable_custom_widget_manager()


def install_runtime_deps(repo_root: Path | str | None = None) -> None:
    enable_colab_widgets()
    if repo_root is not None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(Path(repo_root).resolve())],
            check=True,
        )
    if in_colab():
        subprocess.run(["apt-get", "install", "-qq", "-y", "ffmpeg"], check=True)


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


def resolve_session_config_path(repo_root: Path | str, path: Path | str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(repo_root) / p
    return p.resolve()


def load_repo_session_config(repo_root: Path | str, session_config: Path | str | None = None) -> dict:
    path = resolve_session_config_path(
        repo_root,
        session_config or "configs/ab_prefs.session.json",
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing session config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_provider_list(session: dict) -> list[str]:
    return parse_compare_providers(session.get("compare_providers", ""))


def write_colab_session_config(
    paths: dict[str, Path],
    session_config: Path | str,
    dest: Path | str | None = None,
) -> Path:
    """Write runtime session JSON: settings from session_config, GCS paths from mount."""
    repo = Path(paths["notebook_root"])
    payload = dict(load_repo_session_config(repo, session_config))
    payload.update(
        {
            "notebook_root": str(paths["notebook_root"]),
            "gt_dir": str(paths["gt_dir"]),
            "audio_dir": str(paths["audio_dir"]),
            "config_json": str(paths["config_json"]),
            "session_manifest": str(paths["session_manifest"]),
            "clip_dir": str(paths["clip_dir"]),
            "cache_dir": str(paths["cache_dir"]),
            "output_dir": str(paths["output_dir"]),
        }
    )
    manifest_path = Path(payload["session_manifest"])
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("session_items", "unique_recordings", "recording_seed", "compare_providers", "seed", "min_gt_words", "min_audio_seconds"):
            if key in manifest:
                payload[key] = manifest[key]
    dest = Path(dest or repo / "configs" / "ab_prefs.session.runtime.json")
    if not dest.is_absolute():
        dest = (repo / dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Session config: {resolve_session_config_path(repo, session_config)}")
    print(f"Runtime config: {dest}")
    print(
        f"  session_items={payload.get('session_items')}, "
        f"unique_recordings={payload.get('unique_recordings')}, compare={payload.get('compare_providers')}"
    )
    return dest


def patch_providers_colab(
    asr_root: Path | str,
    providers_path: Path | str,
    provider_names: list[str],
) -> None:
    """Rewrite providers.colab.json for compare pool only."""
    asr_root = Path(asr_root)
    path = Path(providers_path)
    missing = [n for n in provider_names if n not in ASR_PROVIDER_SUBDIRS]
    if missing:
        raise KeyError(f"Unknown compare provider(s): {missing}")
    payload = {"providers": {name: str(asr_root / ASR_PROVIDER_SUBDIRS[name]) for name in provider_names}}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fetch_rater_preferences(
    rater_id: str,
    output_dir: Path,
    bucket: str,
) -> None:
    """Download rater's existing preferences file from GCS if it isn't already local."""
    rater_id = rater_id.strip()
    filename = f"preferences_{rater_id}.json"
    local_path = output_dir / filename
    if local_path.exists():
        print(f"Existing ratings found locally: {local_path}")
        return
    gcs_subdir = output_dir.name          # e.g. "ab_prefs"
    gcs_path = f"gs://{bucket}/{gcs_subdir}/{filename}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["gsutil", "cp", gcs_path, str(local_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        import json
        try:
            n = len(json.loads(local_path.read_text())["responses"])
        except Exception:
            n = "?"
        print(f"Downloaded {n} existing rating(s) for {rater_id} from {gcs_path}")
    else:
        # File not on GCS yet — that's fine, first-time rater
        print(f"No existing ratings on GCS for {rater_id} ({gcs_path}) — starting fresh.")


def bootstrap(
    rater_id: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    mount_point: Path | str = DEFAULT_MOUNT,
    repo_root: Path | str = DEFAULT_REPO,
    repo_url: str = DEFAULT_REPO_URL,
    asr_subdir: str = "asr/dd210",
    session_config: Path | str = "configs/ab_prefs.session.json",
    runtime_session_config: Path | str = "configs/ab_prefs.session.runtime.json",
) -> tuple[Path, Path]:
    """Clone repo, install deps, auth + gcsfuse mount, write runtime session config."""
    enable_colab_widgets()
    repo = clone_repo(repo_root, repo_url=repo_url)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    install_runtime_deps(repo)
    mount = mount_gcs_bucket(bucket, mount_point)
    paths = repo_paths(mount, repo, asr_subdir=asr_subdir)
    base_session = load_repo_session_config(repo, session_config)
    compare = compare_provider_list(base_session)
    patch_providers_colab(paths["asr_root"], paths["config_json"], compare)
    session_path = write_colab_session_config(paths, session_config, dest=runtime_session_config)
    fetch_rater_preferences(rater_id, paths["output_dir"], bucket)
    print(f"GT: {paths['gt_dir']}")
    print(f"Audio: {paths['audio_dir']}")
    print(f"ASR: {paths['asr_root']}")
    print(f"Preferences → {paths['output_dir'] / f'preferences_{rater_id.strip()}.json'}")
    return repo, session_path
