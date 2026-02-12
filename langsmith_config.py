import os
from langsmith import Client
from datetime import datetime
from typing import Dict, Any

class LangSmithMonitor:
    def __init__(self):
        self.client = Client()

    def log_custom_feedback(self, run_id: str, score: float, comment: str=""):
        """Log user feedback on responses """
        self.client.create_feedback(
            run_id = run_id,
            key="user_rating",
            score = score,
            comment = comment
        )
    
    def log_token_usage(self, run_id: str, tokens: int, cost: float):
        """Track token usage and costs"""
        self.client.update_run(
            run_id = run_id,
            extra={
                "tokens_used": tokens,
                "estimated_cost": cost
            }
        )

    def get_project_stats(self, project_name: str)-> Dict[str, Any]:
        """Get aggregated states for project"""
        runs = self.client.list_runs(project_name=project_name)

        total_runs=0
        total_tokens=0
        avg_latency=0
        error_count = 0

        for run in runs:
            total_runs += 1
            if run.extra:
                total_tokens += run.extra.get("tokens_used", 0)
            if run.error:
                error_count += 1
            if run.end_time and run.start_time:
                avg_latency += (run.end_time - run.start_time).total_seconds()

        return {
            "total_runs": total_runs,
            "total_tokens": total_tokens,
            "avg_latency_seconds": avg_latency / total_runs if total_runs else 0,
            "error_rate": error_count / total_runs if total_runs else 0
        }
    
    