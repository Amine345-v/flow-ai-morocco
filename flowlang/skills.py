import os
import json
import glob
from typing import Any, Dict, List, Optional
from loguru import logger


DEFAULT_SKILLS = {
    "zero_circular_imports_tsx": {
        "name": "zero_circular_imports_tsx",
        "description": "TypeScript / React component synthesis policy with isolated interface declarations.",
        "assigned_team": "code_engineers",
        "triggers": ["tsx", "react", "typescript", "component", "import"],
        "procedural_steps": [
            "Export all interface types at the top of the file before component declaration.",
            "Use inline hooks (useState, useEffect) inside functional components.",
            "Ensure zero relative import cycles across module boundaries."
        ],
        "success_rate": 1.0
    },
    "gaap_double_entry_validation": {
        "name": "gaap_double_entry_validation",
        "description": "GAAP double-entry ledger calculation and audit trail validation.",
        "assigned_team": "system_architect",
        "triggers": ["accounting", "ledger", "erp", "gaap", "journal"],
        "procedural_steps": [
            "Verify SUM(credits) == SUM(debits) for every journal transaction.",
            "Attach cryptographically verifiable hash to audit log entries.",
            "Enforce strict decimal precision for financial quantities."
        ],
        "success_rate": 0.98
    }
}


class SkillManager:
    """
    Procedural Skill Extraction & Accumulation Engine for FlowLang / JOL Studio.
    Inspired by Nous Research's Hermes Agent skill learning loop. Automatically distills
    successful task execution patterns into reusable procedural `.flowskill` artifacts.
    """

    def __init__(self, skills_dir: str = "./.flowlang/skills"):
        self.skills_dir = skills_dir
        self.ensure_skills_dir()
        self.skills: Dict[str, Dict[str, Any]] = self.load_all_skills()

    def ensure_skills_dir(self):
        os.makedirs(self.skills_dir, exist_ok=True)
        for skill_id, skill_data in DEFAULT_SKILLS.items():
            path = os.path.join(self.skills_dir, f"{skill_id}.flowskill")
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(skill_data, f, indent=2)

    def load_all_skills(self) -> Dict[str, Dict[str, Any]]:
        """Load all `.flowskill` files from skills directory."""
        loaded = {}
        pattern = os.path.join(self.skills_dir, "*.flowskill")
        for path in glob.glob(pattern):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    skill = json.load(f)
                    name = skill.get("name")
                    if name:
                        loaded[name] = skill
            except Exception as e:
                logger.warning(f"Failed to read skill file '{path}': {e}")
        return loaded

    def create_skill(
        self,
        name: str,
        description: str,
        assigned_team: str,
        procedural_steps: List[str],
        triggers: Optional[List[str]] = None,
        success_rate: float = 1.0
    ) -> str:
        """Create and persist a new procedural skill."""
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        skill_data = {
            "name": safe_name,
            "description": description,
            "assigned_team": assigned_team,
            "triggers": triggers or [safe_name],
            "procedural_steps": procedural_steps,
            "success_rate": success_rate
        }

        path = os.path.join(self.skills_dir, f"{safe_name}.flowskill")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(skill_data, f, indent=2)

        self.skills[safe_name] = skill_data
        logger.info(f"⚡ [SkillManager] Auto-learned skill '{safe_name}' for team '{assigned_team}'")
        return path

    def find_matching_skills(self, team_name: Optional[str] = None, query_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find matching procedural skills based on team role or trigger keywords."""
        matches = []
        clean_query = (query_text or "").lower()

        for skill in self.skills.values():
            if team_name and skill.get("assigned_team") != team_name:
                continue

            triggers = [t.lower() for t in skill.get("triggers", [])]
            if not query_text or any(tr in clean_query for tr in triggers):
                matches.append(skill)

        return matches

    def format_skill_prompt_context(self, team_name: str, query_text: Optional[str] = None) -> str:
        """Formats active procedural skills into markdown context for system prompt injection."""
        matching = self.find_matching_skills(team_name=team_name, query_text=query_text)
        if not matching:
            return ""

        lines = ["\n⚡ [HERMES PROCEDURAL SKILLS & WORKFLOW POLICIES]"]
        for skill in matching:
            lines.append(f"  • Skill: {skill['name'].upper()} (Success Rate: {skill.get('success_rate', 1.0):.0%})")
            lines.append(f"    Description: {skill['description']}")
            lines.append("    Procedural Steps:")
            for step in skill.get("procedural_steps", []):
                lines.append(f"      - {step}")
        lines.append("")
        return "\n".join(lines)
