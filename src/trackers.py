from __future__ import annotations

import os

import requests

from .models import Hypothesis


def configured_trackers() -> dict[str, bool]:
    return {
        "jira": bool(os.getenv("JIRA_BASE_URL") and os.getenv("JIRA_EMAIL") and os.getenv("JIRA_API_TOKEN") and os.getenv("JIRA_PROJECT_KEY")),
        "youtrack": bool(os.getenv("YOUTRACK_BASE_URL") and os.getenv("YOUTRACK_TOKEN") and os.getenv("YOUTRACK_PROJECT_ID")),
    }


def create_jira_issue(hypothesis: Hypothesis) -> str:
    base_url = _required_env("JIRA_BASE_URL").rstrip("/")
    email = _required_env("JIRA_EMAIL")
    token = _required_env("JIRA_API_TOKEN")
    project_key = _required_env("JIRA_PROJECT_KEY")
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": hypothesis.title[:250],
            "description": _jira_description(hypothesis),
            "issuetype": {"name": os.getenv("JIRA_ISSUE_TYPE", "Task")},
            "labels": ["hypothesis-factory", *[tag.replace(" ", "-")[:50] for tag in hypothesis.tags[:5]]],
        }
    }
    response = requests.post(
        f"{base_url}/rest/api/3/issue",
        auth=(email, token),
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    key = response.json().get("key", "")
    return f"{base_url}/browse/{key}" if key else base_url


def create_youtrack_issue(hypothesis: Hypothesis) -> str:
    base_url = _required_env("YOUTRACK_BASE_URL").rstrip("/")
    token = _required_env("YOUTRACK_TOKEN")
    project_id = _required_env("YOUTRACK_PROJECT_ID")
    payload = {
        "project": {"id": project_id},
        "summary": hypothesis.title[:250],
        "description": _plain_description(hypothesis),
    }
    response = requests.post(
        f"{base_url}/api/issues?fields=idReadable",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    issue_id = response.json().get("idReadable", "")
    return f"{base_url}/issue/{issue_id}" if issue_id else base_url


def _jira_description(hypothesis: Hypothesis) -> dict:
    text = _plain_description(hypothesis)
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": text[:30000]}],
            }
        ],
    }


def _plain_description(hypothesis: Hypothesis) -> str:
    sources = "\n".join(f"- {item.source}: {item.quote}" for item in hypothesis.evidence[:5])
    plan = "\n".join(f"- {step}" for step in hypothesis.experiment_plan)
    calculations = "\n".join(f"- {item.name}: {item.status}, {item.value}" for item in hypothesis.calculations)
    return (
        f"{hypothesis.statement}\n\n"
        f"Score: {hypothesis.total_score:.3f}\n"
        f"Mechanism: {hypothesis.mechanism}\n"
        f"Rationale: {hypothesis.rationale}\n\n"
        f"Validation plan:\n{plan}\n\n"
        f"Calculator checks:\n{calculations}\n\n"
        f"Sources:\n{sources}"
    )


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value
