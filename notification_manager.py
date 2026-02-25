"""
Unified notification system: Discord, Email, SMS (stubs).
Autopilot, risk alerts, daily summary.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_notification_prefs() -> Dict[str, Any]:
    try:
        from db import get_setting
        raw = str(get_setting("notification_prefs", "{}") or "{}")
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def notify(
    event: str,
    payload: Dict[str, Any],
    force_discord: bool = False,
) -> bool:
    """
    Send notification for event. Events: autopilot_bot_created, autopilot_bot_closed,
    risk_alert, daily_summary, maintenance_mode.
    """
    prefs = _get_notification_prefs()
    if not prefs.get("enabled", True) and not force_discord:
        return False
    sent = False
    if prefs.get("discord", True) or force_discord:
        sent = _notify_discord(event, payload) or sent
    if prefs.get("email", False):
        _notify_email_stub(event, payload)
    if prefs.get("sms", False):
        _notify_sms_stub(event, payload)
    return sent


def _notify_discord(event: str, payload: Dict[str, Any]) -> bool:
    try:
        from discord_notifications import DiscordNotifier
        url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        if not url:
            return False
        notifier = DiscordNotifier()
        if event == "autopilot_bot_created":
            msg = f"🤖 **Autopilot**: Created bot for {payload.get('symbol', '?')} (score {payload.get('score', 0):.0f})"
        elif event == "autopilot_bot_closed":
            msg = f"⏹️ **Autopilot**: Closed bot for {payload.get('symbol', '?')} (score dropped to {payload.get('score', 0):.0f})"
        elif event == "risk_alert":
            msg = f"⚠️ **Risk Alert**: {payload.get('message', '')}"
        elif event == "daily_summary":
            msg = f"📊 **Daily Summary**: {payload.get('message', '')}"
        elif event == "maintenance_mode":
            msg = f"🔧 **Maintenance**: {payload.get('message', '')}"
        else:
            msg = f"📌 **{event}**: {json.dumps(payload)[:200]}"
        notifier.send_message(msg, force=payload.get("force", False))
        return True
    except Exception as e:
        logger.debug("Discord notify failed: %s", e)
        return False


def _notify_email_stub(event: str, payload: Dict[str, Any]) -> None:
    """Stub for email. Implement via SendGrid/SES when needed."""
    logger.info("Email stub: event=%s payload=%s", event, payload)


def _notify_sms_stub(event: str, payload: Dict[str, Any]) -> None:
    """Stub for SMS. Implement via Twilio when needed."""
    logger.info("SMS stub: event=%s payload=%s", event, payload)
