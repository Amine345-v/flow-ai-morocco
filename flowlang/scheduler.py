import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from loguru import logger


class FlowSchedulerJob:
    """Represents a scheduled FlowLang execution job."""

    def __init__(self, job_id: str, flow_name: str, cron_expr: str, callback: Callable[[], Any]):
        self.job_id = job_id
        self.flow_name = flow_name
        self.cron_expr = cron_expr
        self.callback = callback
        self.interval_s = self._parse_cron(cron_expr)
        self.last_run: Optional[datetime] = None
        self.run_count = 0
        self.status = "PENDING"
        self.last_error: Optional[str] = None

    def _parse_cron(self, cron_expr: str) -> int:
        """Parse simple cron tokens or integer seconds."""
        expr = cron_expr.strip().lower()
        if expr == "@hourly":
            return 3600
        elif expr == "@daily":
            return 86400
        elif expr == "@minutely" or expr == "@every_1m":
            return 60
        elif expr == "@every_5s":
            return 5
        elif expr.isdigit():
            return int(expr)
        return 3600  # Default 1 hour fallback

    def should_run(self) -> bool:
        if self.status == "RUNNING":
            return False
        if not self.last_run:
            return True
        elapsed = (datetime.now() - self.last_run).total_seconds()
        return elapsed >= self.interval_s

    def execute(self):
        self.status = "RUNNING"
        self.last_run = datetime.now()
        self.run_count += 1
        logger.info(f"⏰ [FlowScheduler] Triggering job '{self.job_id}' for flow '{self.flow_name}' (Run #{self.run_count})")
        try:
            self.callback()
            self.status = "COMPLETED"
            self.last_error = None
        except Exception as e:
            self.status = "FAILED"
            self.last_error = str(e)
            logger.error(f"❌ [FlowScheduler] Job '{self.job_id}' failed: {e}")


class FlowScheduler:
    """
    Background Cron & Scheduled Flow Daemon inspired by Hermes Agent cron scheduler.
    Performs unattended scheduled software factory runs and security audits.
    """

    def __init__(self, poll_interval_s: float = 1.0):
        self.jobs: Dict[str, FlowSchedulerJob] = {}
        self.poll_interval_s = poll_interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_job(self, job_id: str, flow_name: str, cron_expr: str, callback: Callable[[], Any]) -> FlowSchedulerJob:
        """Register a scheduled flow job."""
        job = FlowSchedulerJob(job_id, flow_name, cron_expr, callback)
        self.jobs[job_id] = job
        logger.info(f"📅 [FlowScheduler] Registered job '{job_id}' for flow '{flow_name}' ({cron_expr})")
        return job

    def start(self):
        """Start the background scheduler daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("🚀 [FlowScheduler] Daemon thread started.")

    def _loop(self):
        while self._running:
            for job in list(self.jobs.values()):
                if job.should_run():
                    threading.Thread(target=job.execute, daemon=True).start()
            time.sleep(self.poll_interval_s)

    def stop(self):
        """Stop background scheduler daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("🛑 [FlowScheduler] Daemon thread stopped.")

    def get_job_statuses(self) -> List[Dict[str, Any]]:
        """Return status telemetry for all scheduled jobs."""
        return [
            {
                "job_id": j.job_id,
                "flow_name": j.flow_name,
                "cron_expr": j.cron_expr,
                "status": j.status,
                "run_count": j.run_count,
                "last_run": j.last_run.isoformat() if j.last_run else None,
                "last_error": j.last_error
            }
            for j in self.jobs.values()
        ]
