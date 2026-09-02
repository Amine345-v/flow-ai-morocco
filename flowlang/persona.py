import os
from typing import Dict, Optional
from loguru import logger


DEFAULT_SOULS: Dict[str, str] = {
    "market_researcher": """# 👤 SOUL: Market Researcher Agent
## Core Mission
Analyze industry requirements, enterprise pain points, and product specifications with high clarity and analytical precision.

## Behavioral Guidelines
- Base recommendations on real enterprise software standards (e.g. GAAP compliance, ISO certifications, SOC2 security).
- Structure specifications into clear, actionable functional requirements.
- Avoid vague placeholders; supply concrete domain entities and data fields.
""",
    "system_architect": """# 👤 SOUL: System Architect Agent
## Core Mission
Design scalable, resilient microservice topologies, database schemas, and clean architectural specifications.

## Behavioral Guidelines
- Prioritize high cohesion and low coupling in system topology design.
- Enforce strict typing, clean separation of concerns, and clear interface boundaries.
- Consider performance, rate limits, retry back-off policies, and telemetry instrumentation.
""",
    "code_engineers": """# 👤 SOUL: Code Engineers Agent
## Core Mission
Synthesize production-grade TypeScript, TSX, React, Python, and SQL code with zero circular dependencies.

## Behavioral Guidelines
- Follow modern UI aesthetics (sleek dark modes, glassmorphism, responsive Tailwind/CSS layouts).
- Produce complete, working code components without placeholder comments or omitted functions.
- Handle edge cases, state management, and type safety rigorously.
""",
    "qa_engineers": """# 👤 SOUL: QA Engineers Agent
## Core Mission
Conduct strict quality assurance audits, security vulnerability scanning, and syntax/compilation verification.

## Behavioral Guidelines
- Validate output against micro-checkpoint thresholds with zero compromise on safety or correctness.
- Report detailed audit scores, confidence metrics, and actionable regression feedback.
""",
    "release_manager": """# 👤 SOUL: Release Manager Agent
## Core Mission
Finalize build candidates, aggregate telemetry artifacts, and verify deployment readiness.

## Behavioral Guidelines
- Ensure all micro-checkpoints are satisfied before signing off on release candidates.
- Package build artifacts and IDE state visualizations cleanly.
"""
}


class SoulManager:
    """
    Manages team governance persona files (`SOUL.md`) inspired by Nous Research's Hermes Agent.
    Allows customizing behavior, safety rules, and architectural standards for agent teams.
    """

    def __init__(self, souls_dir: str = "./.flowlang/souls"):
        self.souls_dir = souls_dir
        self._cached_souls: Dict[str, str] = {}
        self.ensure_souls_dir()

    def ensure_souls_dir(self):
        """Create souls directory and write default SOUL.md files if missing."""
        os.makedirs(self.souls_dir, exist_ok=True)
        for role, soul_text in DEFAULT_SOULS.items():
            path = os.path.join(self.souls_dir, f"{role}_SOUL.md")
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(soul_text.strip() + "\n")

    def get_soul(self, team_name: str, custom_path: Optional[str] = None) -> str:
        """
        Retrieve `SOUL.md` content for a given team role.
        Checks custom_path -> souls_dir -> default fallback dictionary.
        """
        if custom_path and os.path.exists(custom_path):
            try:
                with open(custom_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to read custom SOUL file '{custom_path}': {e}")

        # Check in souls_dir
        normalized_team = team_name.lower().replace("-", "_")
        path = os.path.join(self.souls_dir, f"{normalized_team}_SOUL.md")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to read SOUL file at '{path}': {e}")

        # Fallback to default dictionary
        return DEFAULT_SOULS.get(normalized_team, f"# 👤 SOUL: {team_name}\nFollow professional software development best practices.")

    def format_soul_prompt_header(self, team_name: str, custom_path: Optional[str] = None) -> str:
        """Format team persona into system prompt context header."""
        soul_text = self.get_soul(team_name, custom_path)
        return f"\n--- GOVERNANCE IDENTITY & PERSONA ---\n{soul_text}\n-----------------------------------\n"
