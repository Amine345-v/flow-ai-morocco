import os
import json
import uuid
import concurrent.futures
from typing import Any, Dict, List, Optional
from loguru import logger

from .ai_providers import select_provider
from .persona import SoulManager
from .memory import HermesMemoryStore


class SubAgent:
    """
    Lightweight Sub-Agent instance spawned by a parent agent flow.
    Operates with an isolated task description, custom persona/SOUL, and scoped memory.
    Inspired by Nous Research Hermes Agent sub-agent delegation.
    """

    def __init__(
        self,
        subagent_id: str,
        role_name: str,
        task_description: str,
        parent_id: Optional[str] = None,
        custom_soul: Optional[str] = None
    ):
        self.subagent_id = subagent_id or f"subagent_{uuid.uuid4().hex[:8]}"
        self.role_name = role_name
        self.task_description = task_description
        self.parent_id = parent_id
        self.custom_soul = custom_soul
        self.status = "CREATED"
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def execute(self, flow_name: str = "subagent_flow", use_mock: bool = False) -> Dict[str, Any]:
        """Execute the subagent task autonomously via AI Provider."""
        self.status = "RUNNING"
        logger.info(f"🤖 [SubAgent] Spawning child '{self.subagent_id}' [{self.role_name}] for task: '{self.task_description[:60]}...'")

        if use_mock:
            self.result = {
                "subagent_id": self.subagent_id,
                "role": self.role_name,
                "summary": f"Completed sub-task: {self.task_description[:50]}",
                "output": f"Simulated output artifact for role {self.role_name}",
                "artifacts": ["subagent_result.json"]
            }
            self.status = "COMPLETED"
            logger.info(f"✅ [SubAgent] Child '{self.subagent_id}' completed (mock).")
            return self.result

        prompt = (
            f"SUB-AGENT DELEGATION TASK\n"
            f"Sub-Agent ID: {self.subagent_id}\n"
            f"Role: {self.role_name}\n"
            f"Parent Execution ID: {self.parent_id or 'Root'}\n"
            f"Task:\n{self.task_description}\n\n"
            f"Please complete this specific sub-task and return JSON response: {{\n"
            f"  \"summary\": string,\n"
            f"  \"output\": string,\n"
            f"  \"artifacts\": list of strings\n"
            f"}}"
        )

        try:
            ai = select_provider()
            resp = ai.execute(
                self.role_name,
                "try",
                [prompt],
                {"flow_name": flow_name, "query_text": self.task_description}
            )
            raw_output = resp.get("output", "")
            clean = raw_output
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()

            try:
                parsed = json.loads(clean)
            except Exception:
                parsed = {"summary": "Task completed.", "output": raw_output, "artifacts": []}

            self.result = {
                "subagent_id": self.subagent_id,
                "role": self.role_name,
                "summary": parsed.get("summary", "Task executed successfully."),
                "output": parsed.get("output", raw_output),
                "artifacts": parsed.get("artifacts", [])
            }
            self.status = "COMPLETED"
            logger.info(f"✅ [SubAgent] Child '{self.subagent_id}' completed successfully.")
            return self.result
        except Exception as ex:
            self.status = "FAILED"
            self.error = str(ex)
            logger.error(f"❌ [SubAgent] Child '{self.subagent_id}' failed: {ex}")
            return {
                "subagent_id": self.subagent_id,
                "role": self.role_name,
                "status": "FAILED",
                "error": str(ex)
            }


class SubAgentOrchestrator:
    """
    Manages spawning, parallel swarm execution, and aggregation of Sub-Agent tasks.
    """

    def __init__(self):
        self.subagents: Dict[str, SubAgent] = {}

    def spawn(
        self,
        role_name: str,
        task_description: str,
        parent_id: Optional[str] = None,
        custom_soul: Optional[str] = None
    ) -> SubAgent:
        sub_id = f"subagent_{uuid.uuid4().hex[:8]}"
        sa = SubAgent(
            subagent_id=sub_id,
            role_name=role_name,
            task_description=task_description,
            parent_id=parent_id,
            custom_soul=custom_soul
        )
        self.subagents[sub_id] = sa
        return sa

    def execute_swarm(self, subagent_ids: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
        """Run multiple sub-agents in parallel using a thread pool."""
        logger.info(f"🐝 [SubAgentOrchestrator] Executing swarm of {len(subagent_ids)} sub-agents...")
        results = []
        targets = [self.subagents[sid] for sid in subagent_ids if sid in self.subagents]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sa = {executor.submit(sa.execute): sa for sa in targets}
            for future in concurrent.futures.as_completed(future_to_sa):
                sa = future_to_sa[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as ex:
                    results.append({
                        "subagent_id": sa.subagent_id,
                        "role": sa.role_name,
                        "status": "FAILED",
                        "error": str(ex)
                    })

        return results
