#!/usr/bin/env python3
"""Weekly index update scheduler."""

# OPTIMIZED_BY_CODEX_STEP_3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.schemas import BuildIndexRequest
from app.services.index_service import IndexService
from app.storage.persistence import PersistenceManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weekly PaperRAG index update")
    parser.add_argument("--data", type=str, default="", help="Data file path for metadata ingestion")
    parser.add_argument("--max_papers", type=int, default=20000, help="Maximum metadata records per update")
    parser.add_argument("--day", type=str, default=os.getenv("PAPERRAG_SCHEDULE_DAY_OF_WEEK", "sun"))
    parser.add_argument("--hour", type=int, default=int(os.getenv("PAPERRAG_SCHEDULE_HOUR", "3")))
    parser.add_argument("--minute", type=int, default=int(os.getenv("PAPERRAG_SCHEDULE_MINUTE", "0")))
    parser.add_argument(
        "--timezone",
        type=str,
        default=os.getenv("PAPERRAG_SCHEDULE_TIMEZONE", "Asia/Shanghai"),
    )
    parser.add_argument("--run_now", action="store_true", help="Run once immediately before scheduling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError as exc:
        raise RuntimeError("APScheduler is required. Run: pip install -r requirements.txt") from exc

    setup_logging()
    settings = get_settings()

    persistence = PersistenceManager(settings)
    service = IndexService(settings, persistence)

    data_path = args.data or str(settings.abs_data_dir / settings.default_data_file)

    def _job() -> None:
        request = BuildIndexRequest(
            data_path=data_path,
            limit=args.max_papers,
            rebuild=True,
        )
        result = service.build(request)
        print(
            f"[weekly-update] status={result.status} docs={result.num_documents} "
            f"chunks={result.num_chunks} elapsed_ms={result.elapsed_ms:.0f}"
        )

    scheduler = BlockingScheduler(timezone=args.timezone)
    scheduler.add_job(
        _job,
        trigger="cron",
        day_of_week=args.day,
        hour=args.hour,
        minute=args.minute,
        id="paperrag_weekly_update",
        replace_existing=True,
    )

    if args.run_now:
        _job()

    print(
        f"Scheduler started: day={args.day} hour={args.hour} minute={args.minute} "
        f"timezone={args.timezone} max_papers={args.max_papers}"
    )
    scheduler.start()


if __name__ == "__main__":
    main()
# STEP_3_SUMMARY: Added APScheduler-based weekly index refresh script with PAPERRAG_ schedule env support.
