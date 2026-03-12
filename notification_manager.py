"""
Unified notification system: Discord, Email, SMS (stubs), Telegram, Browser Push.
Autopilot, risk alerts, daily summary, trade notifications.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

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


_last_gate_block_by_symbol: Dict[str, Tuple[float, str]] = {}
_GATE_BLOCK_COOLDOWN_SEC = 600  # rate-limit gate-blocked notifications per symbol

def _notify_discord(event: str, payload: Dict[str, Any]) -> bool:
    try:
        from discord_notifications import DiscordNotifier
        url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        if not url:
            return False
        notifier = DiscordNotifier()

        if event == "autopilot_bot_created":
            sym = payload.get("symbol", "?")
            score = payload.get("score", 0)
            entry_type = payload.get("entry_type", "")
            confidence = payload.get("confidence", 0)
            evidence = payload.get("evidence", [])
            targets = payload.get("target_levels", {})
            invalidation = payload.get("invalidation_level", 0)

            msg = f"**Autopilot**: Created bot for **{sym}** (score {score:.0f})"
            if entry_type:
                msg += f"\nEntry: {entry_type} | Confidence: {confidence:.0%}"
            if targets.get("tp1"):
                msg += f"\nTP1: {targets['tp1']:.4f} | SL: {invalidation:.4f}"
            if evidence:
                msg += "\n" + "\n".join(f"- {e}" for e in evidence[:3])

        elif event == "autopilot_bot_closed":
            msg = f"**Autopilot**: Closed bot for {payload.get('symbol', '?')} (score dropped to {payload.get('score', 0):.0f})"

        elif event == "gate_blocked":
            sym = payload.get("symbol", "?")
            reason = payload.get("reason", "unknown")
            now = time.time()
            last_ts, last_reason = _last_gate_block_by_symbol.get(sym, (0, ""))
            if (now - last_ts) < _GATE_BLOCK_COOLDOWN_SEC and last_reason == reason:
                return False
            _last_gate_block_by_symbol[sym] = (now, reason)
            msg = f"**Gate Blocked**: {sym} — {reason}"

        elif event == "watchlist_triggered":
            sym = payload.get("symbol", "?")
            msg = f"**Watchlist**: {sym} now READY (confidence {payload.get('confidence', 0):.0%}, {payload.get('entry_type', '')})"

        elif event == "risk_alert":
            msg = f"**Risk Alert**: {payload.get('message', '')}"
        elif event == "daily_summary":
            msg = f"**Daily Summary**: {payload.get('message', '')}"
        elif event == "maintenance_mode":
            msg = f"**Maintenance**: {payload.get('message', '')}"
        else:
            msg = f"**{event}**: {json.dumps(payload)[:200]}"
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


# =========================================================
# Discord Webhook Notifications
# =========================================================
def send_discord_notification(
    webhook_url: str,
    title: str,
    message: str,
    color: int = 0x00ff00,
    fields: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """
    Send a Discord embed via webhook.
    color: 0x00ff00 (green) for profits, 0xff0000 (red) for losses,
           0xffa500 (orange) for warnings, 0x3b82f6 (blue) for info
    """
    try:
        if not webhook_url or not webhook_url.strip():
            return False

        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if fields:
            embed["fields"] = fields

        payload = {"embeds": [embed]}

        response = requests.post(
            webhook_url.strip(),
            json=payload,
            timeout=5
        )
        return response.status_code in (200, 204)
    except Exception as e:
        logger.debug("Discord webhook failed: %s", e)
        return False


# =========================================================
# Telegram Bot Notifications
# =========================================================
def send_telegram_notification(
    bot_token: str,
    chat_id: str,
    message: str,
) -> bool:
    """
    Send a message via Telegram Bot API.
    API endpoint: https://api.telegram.org/bot{token}/sendMessage
    """
    try:
        if not bot_token or not bot_token.strip() or not chat_id or not chat_id.strip():
            return False

        url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
        payload = {
            "chat_id": chat_id.strip(),
            "text": message,
            "parse_mode": "HTML",
        }

        response = requests.post(url, json=payload, timeout=5)
        return response.status_code in (200, 201)
    except Exception as e:
        logger.debug("Telegram notification failed: %s", e)
        return False


# =========================================================
# Notification Storage & Retrieval
# =========================================================
def insert_notification(
    notification_type: str,
    title: str,
    message: str,
    bot_id: Optional[int] = None,
) -> int:
    """Insert notification into database. Returns notification_id."""
    try:
        from db import _conn, now_ts
        con = _conn()
        con.execute(
            """
            INSERT INTO notifications(timestamp, type, title, message, bot_id, read)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (now_ts(), notification_type, title, message, bot_id),
        )
        con.commit()
        result = con.execute("SELECT last_insert_rowid() as id").fetchone()
        con.close()
        return result["id"] if result else -1
    except Exception as e:
        logger.debug("Failed to insert notification: %s", e)
        return -1


def get_notifications(limit: int = 50, unread_only: bool = False) -> List[Dict[str, Any]]:
    """Get recent notifications from database."""
    try:
        from db import _conn
        con = _conn()
        query = "SELECT * FROM notifications"
        params = []

        if unread_only:
            query += " WHERE read = 0"

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = con.execute(query, params).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug("Failed to get notifications: %s", e)
        return []


def mark_notification_read(notification_id: int) -> bool:
    """Mark a notification as read."""
    try:
        from db import _conn
        con = _conn()
        con.execute("UPDATE notifications SET read = 1 WHERE id = ?", (notification_id,))
        con.commit()
        con.close()
        return True
    except Exception as e:
        logger.debug("Failed to mark notification read: %s", e)
        return False


def get_unread_count() -> int:
    """Get count of unread notifications."""
    try:
        from db import _conn
        con = _conn()
        result = con.execute("SELECT COUNT(*) as count FROM notifications WHERE read = 0").fetchone()
        con.close()
        return result["count"] if result else 0
    except Exception as e:
        logger.debug("Failed to get unread count: %s", e)
        return 0


# =========================================================
# Notification Event Functions
# =========================================================
def notify_trade_executed(
    bot_name: str,
    symbol: str,
    side: str,
    amount: float,
    price: float,
) -> bool:
    """Notify when a trade is executed."""
    try:
        from db import get_setting, _conn

        title = f"Trade Executed: {symbol}"
        message = f"**{bot_name}** {side.upper()} {amount} @ {price:.4f}"

        # Store in DB
        con = _conn()
        con.execute(
            """
            INSERT INTO notifications(timestamp, type, title, message, bot_id, read)
            VALUES (?, ?, ?, ?, NULL, 0)
            """,
            (int(time.time()), "trade_executed", title, message),
        )
        con.commit()
        con.close()

        # Send to Discord if configured
        webhook_url = get_setting("discord_webhook_url", "").strip()
        if webhook_url:
            color = 0x089981 if side.lower() == "buy" else 0xf23645
            send_discord_notification(
                webhook_url,
                title,
                message,
                color=color,
            )

        # Send to Telegram if configured
        bot_token = get_setting("telegram_bot_token", "").strip()
        chat_id = get_setting("telegram_chat_id", "").strip()
        if bot_token and chat_id:
            send_telegram_notification(bot_token, chat_id, f"{title}\n{message}")

        return True
    except Exception as e:
        logger.debug("notify_trade_executed failed: %s", e)
        return False


def notify_take_profit(
    bot_name: str,
    symbol: str,
    profit_amount: float,
    profit_pct: float,
) -> bool:
    """Notify when take profit is hit."""
    try:
        from db import get_setting, _conn

        title = f"Take Profit: {symbol}"
        message = f"**{bot_name}** closed profitably: +{profit_amount:.2f} ({profit_pct:.2%})"

        con = _conn()
        con.execute(
            """
            INSERT INTO notifications(timestamp, type, title, message, bot_id, read)
            VALUES (?, ?, ?, ?, NULL, 0)
            """,
            (int(time.time()), "take_profit", title, message),
        )
        con.commit()
        con.close()

        webhook_url = get_setting("discord_webhook_url", "").strip()
        if webhook_url:
            send_discord_notification(
                webhook_url,
                title,
                message,
                color=0x00ff00,  # Green for profit
            )

        bot_token = get_setting("telegram_bot_token", "").strip()
        chat_id = get_setting("telegram_chat_id", "").strip()
        if bot_token and chat_id:
            send_telegram_notification(bot_token, chat_id, f"{title}\n{message}")

        return True
    except Exception as e:
        logger.debug("notify_take_profit failed: %s", e)
        return False


def notify_stop_loss(
    bot_name: str,
    symbol: str,
    loss_amount: float,
    loss_pct: float,
) -> bool:
    """Notify when stop loss is hit."""
    try:
        from db import get_setting, _conn

        title = f"Stop Loss: {symbol}"
        message = f"**{bot_name}** hit stop loss: {loss_amount:.2f} ({loss_pct:.2%})"

        con = _conn()
        con.execute(
            """
            INSERT INTO notifications(timestamp, type, title, message, bot_id, read)
            VALUES (?, ?, ?, ?, NULL, 0)
            """,
            (int(time.time()), "stop_loss", title, message),
        )
        con.commit()
        con.close()

        webhook_url = get_setting("discord_webhook_url", "").strip()
        if webhook_url:
            send_discord_notification(
                webhook_url,
                title,
                message,
                color=0xff0000,  # Red for loss
            )

        bot_token = get_setting("telegram_bot_token", "").strip()
        chat_id = get_setting("telegram_chat_id", "").strip()
        if bot_token and chat_id:
            send_telegram_notification(bot_token, chat_id, f"{title}\n{message}")

        return True
    except Exception as e:
        logger.debug("notify_stop_loss failed: %s", e)
        return False


def notify_bot_error(bot_name: str, error_message: str) -> bool:
    """Notify when a bot encounters an error."""
    try:
        from db import get_setting, _conn

        title = f"Bot Error: {bot_name}"
        message = f"Error: {error_message[:200]}"

        con = _conn()
        con.execute(
            """
            INSERT INTO notifications(timestamp, type, title, message, bot_id, read)
            VALUES (?, ?, ?, ?, NULL, 0)
            """,
            (int(time.time()), "bot_error", title, message),
        )
        con.commit()
        con.close()

        webhook_url = get_setting("discord_webhook_url", "").strip()
        if webhook_url:
            send_discord_notification(
                webhook_url,
                title,
                message,
                color=0xff0000,  # Red for error
            )

        bot_token = get_setting("telegram_bot_token", "").strip()
        chat_id = get_setting("telegram_chat_id", "").strip()
        if bot_token and chat_id:
            send_telegram_notification(bot_token, chat_id, f"{title}\n{message}")

        return True
    except Exception as e:
        logger.debug("notify_bot_error failed: %s", e)
        return False


def notify_drawdown_alert(portfolio_pnl_pct: float) -> bool:
    """Notify when portfolio drawdown exceeds threshold."""
    try:
        from db import get_setting, _conn

        title = "Drawdown Alert"
        message = f"Portfolio drawdown: {portfolio_pnl_pct:.2%}"

        con = _conn()
        con.execute(
            """
            INSERT INTO notifications(timestamp, type, title, message, bot_id, read)
            VALUES (?, ?, ?, ?, NULL, 0)
            """,
            (int(time.time()), "drawdown_alert", title, message),
        )
        con.commit()
        con.close()

        webhook_url = get_setting("discord_webhook_url", "").strip()
        if webhook_url:
            send_discord_notification(
                webhook_url,
                title,
                message,
                color=0xffa500,  # Orange for warning
            )

        bot_token = get_setting("telegram_bot_token", "").strip()
        chat_id = get_setting("telegram_chat_id", "").strip()
        if bot_token and chat_id:
            send_telegram_notification(bot_token, chat_id, f"{title}\n{message}")

        return True
    except Exception as e:
        logger.debug("notify_drawdown_alert failed: %s", e)
        return False


def notify_daily_summary(
    total_pnl: float,
    win_count: int,
    loss_count: int,
    best_trade: str,
    worst_trade: str,
) -> bool:
    """Notify with daily trading summary."""
    try:
        from db import get_setting, _conn

        title = "Daily Summary"
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        message = f"Trades: {total_trades} | Wins: {win_count} ({win_rate:.0f}%) | P&L: {total_pnl:.2f}\nBest: {best_trade} | Worst: {worst_trade}"

        con = _conn()
        con.execute(
            """
            INSERT INTO notifications(timestamp, type, title, message, bot_id, read)
            VALUES (?, ?, ?, ?, NULL, 0)
            """,
            (int(time.time()), "daily_summary", title, message),
        )
        con.commit()
        con.close()

        webhook_url = get_setting("discord_webhook_url", "").strip()
        if webhook_url:
            color = 0x00ff00 if total_pnl >= 0 else 0xff0000
            send_discord_notification(
                webhook_url,
                title,
                message,
                color=color,
            )

        bot_token = get_setting("telegram_bot_token", "").strip()
        chat_id = get_setting("telegram_chat_id", "").strip()
        if bot_token and chat_id:
            send_telegram_notification(bot_token, chat_id, f"{title}\n{message}")

        return True
    except Exception as e:
        logger.debug("notify_daily_summary failed: %s", e)
        return False
