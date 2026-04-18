"""Dashboard page.

Shows the status of the train + evaluate pipeline: dataset presence,
trained model count, and any active task. No collection / registry UI.
"""

import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st

app_dir = Path(__file__).parent.parent
project_root = app_dir.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))
if str(project_root / "scripts" / "data") not in sys.path:
    sys.path.insert(0, str(project_root / "scripts" / "data"))

from components.common_sidebar import render_common_sidebar
from services.path_coordinator import (
    DATASETS_DIR,
    FINETUNED_DIR,
    PYBULLET_HSR_ANNOTATION_ROOT,
)


def show_dashboard_page() -> None:
    render_common_sidebar()

    st.title("📊 Dashboard")

    coordinator = st.session_state.path_coordinator
    task_manager = st.session_state.task_manager

    datasets = coordinator.get_datasets()
    trained = coordinator.get_trained_models()
    active_tasks = task_manager.get_active_tasks()

    _render_counts(datasets, trained, active_tasks)
    st.markdown("---")
    _render_sync_state(coordinator)
    st.markdown("---")
    _render_pybullet_status()
    st.markdown("---")
    _render_datasets(datasets)
    st.markdown("---")
    _render_trained(trained)


def _render_sync_state(coordinator) -> None:
    st.subheader("Dataset sync")
    state = coordinator.get_latest_sync_state()
    if state["state"] == "no_dumps":
        st.info("No pybullet_hsr dump found. Generate one under "
                f"`{PYBULLET_HSR_ANNOTATION_ROOT}` first.")
        return

    dump = state["dump"]
    dump_name = dump["path"].name
    dataset_name = state["local_dataset_name"]

    if state["state"] == "up_to_date":
        st.success(
            f"`datasets/{dataset_name}` is in sync with the newest dump `{dump_name}`."
        )
        return

    if state["state"] == "no_local":
        st.warning(
            f"Newest dump `{dump_name}` has not been prepared yet "
            f"(`datasets/{dataset_name}/` missing)."
        )
    else:
        st.warning(
            f"`datasets/{dataset_name}` is stale vs newest dump `{dump_name}` "
            f"— {state['reason']}."
        )

    if st.button("🔄 Sync latest dump now", type="primary"):
        _run_sync(dump["path"])


def _run_sync(dump_path) -> None:
    from prepare_dataset import ensure_dataset
    with st.spinner(f"Preparing {dump_path.name}…"):
        try:
            result = ensure_dataset(source=dump_path, symlink=True)
        except SystemExit as exc:
            st.error(f"prepare failed: {exc}")
            return
        except Exception as exc:  # pragma: no cover - UI safety net
            st.error(f"prepare failed: {exc}")
            return
    counts = result.get("counts", {})
    if result["action"] == "up-to-date":
        st.info("Already up-to-date.")
    else:
        st.success(
            f"Prepared `datasets/{result['dataset_name']}` "
            f"(train={counts.get('train', 0)}, val={counts.get('val', 0)})."
        )
    st.cache_data.clear()
    st.rerun()


def _render_counts(datasets: List[Dict], trained: List[Dict], active: List) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Datasets", len(datasets))
    c2.metric("Trained models", len(trained))
    c3.metric("Active tasks", len(active))


def _render_pybullet_status() -> None:
    st.subheader("pybullet_hsr dumps")
    if not PYBULLET_HSR_ANNOTATION_ROOT.is_dir():
        st.warning(
            f"`{PYBULLET_HSR_ANNOTATION_ROOT}` not found. "
            "Set `PYBULLET_HSR_ROOT` to point at the pybullet_hsr clone."
        )
        return

    coordinator = st.session_state.path_coordinator
    dumps = coordinator.get_available_pybullet_hsr_dumps()
    if not dumps:
        st.warning(
            f"No manifest-bearing dumps under `{PYBULLET_HSR_ANNOTATION_ROOT}`. "
            "Run `pybullet_hsr/scripts/write_manifest.py --dump-dir <dir>` on each dump."
        )
        return

    rows = []
    for d in dumps:
        m = d["manifest"]
        rows.append({
            "Dump": d["path"].name,
            "Dataset": m.get("dataset_name", "?"),
            "Created": m.get("created_at", "")[:19],
            "Format": m.get("label_format", "?"),
            "Classes": len(m.get("classes", [])),
            "Images": m.get("stats", {}).get("num_images", "?"),
        })
    try:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except ImportError:
        for row in rows:
            st.write(
                f"- `{row['Dump']}` — {row['Dataset']} "
                f"({row['Classes']} cls, {row['Images']} imgs, {row['Format']})"
            )


def _render_datasets(datasets: List[Dict]) -> None:
    st.subheader("Local datasets")
    if not datasets:
        st.info(
            "No dataset found under "
            f"`{DATASETS_DIR}`. Run `scripts/data/prepare_dataset.py` or use the Training page."
        )
        return
    for d in datasets[:10]:
        st.write(f"- `{d['name']}` ({d['created']})")


def _render_trained(trained: List[Dict]) -> None:
    st.subheader("Trained models")
    if not trained:
        st.info(f"No trained models in `{FINETUNED_DIR}`.")
        return
    for m in trained[:10]:
        best = m.get("best_path") or m.get("last_path") or "-"
        st.write(f"- `{m['name']}` ({m['created']}) → {best}")


if __name__ == "__main__":
    show_dashboard_page()
