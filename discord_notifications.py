"""
Enhanced Discord Notifications - Full lifecycle coverage for trading bot.

Features:
- Trade signals detected
- Bot deployment/entry
- Trade updates (TP hit, stop moved)
- Trade closure (profit/loss)
- Strategy switches
- Daily summary
- Error alerts
- Portfolio risk alerts

Based on: "Upgrading to an Autonomous Multi-Asset Trading Bot" specification.
"""

import os
import time
import json
import logging
import threading
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """Configuration for Discord notifications."""
    webhook_url: str = ""
    status_webhook_url: str = ""  # Optional separate channel for status
    
    # What to notify
    notify_signals: bool = True
    notify_entries: bool = True
    notify_updates: bool = True
    notify_exits: bool = True
    notify_strategy_changes: bool = True
    notify_risk_alerts: bool = True
    notify_daily_summary: bool = True
    notify_errors: bool = True
    
    # Rate limiting
    min_interval_seconds: int = 5
    max_per_minute: int = 20
    
    # Formatting
    include_reasoning: bool = True
    include_confidence: bool = True
    compact_mode: bool = False


class DiscordNotifier:
    """
    Enhanced Discord notification system.
    
    Handles all types of trading notifications with rate limiting,
    message queuing, and formatting.
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig(
            webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
            status_webhook_url=os.getenv("DISCORD_STATUS_WEBHOOK_URL", "")
        )
        
        self._last_send_time = 0
        self._send_count_minute = 0
        self._minute_start = 0
        self._lock = threading.Lock()
        self._message_queue: deque = deque(maxlen=100)
        
        # Track daily stats for summary
        self._daily_stats = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "trades_opened": 0,
            "trades_closed": 0,
            "total_pnl": 0.0,
            "winners": 0,
            "losers": 0,
            "signals_detected": 0,
            "alerts_sent": 0
        }
        
        logger.info(f"DiscordNotifier initialized (webhook configured: {bool(self.config.webhook_url)})")
    
    def _reset_daily_stats(self) -> None:
        """Reset daily stats at day boundary."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_stats["date"]:
            self._daily_stats = {
                "date": today,
                "trades_opened": 0,
                "trades_closed": 0,
                "total_pnl": 0.0,
                "winners": 0,
                "losers": 0,
                "signals_detected": 0,
                "alerts_sent": 0
            }
    
    def _can_send(self) -> bool:
        """Check rate limits."""
        now = time.time()
        
        # Reset minute counter
        if now - self._minute_start > 60:
            self._send_count_minute = 0
            self._minute_start = now
        
        # Check limits
        if now - self._last_send_time < self.config.min_interval_seconds:
            return False
        
        if self._send_count_minute >= self.config.max_per_minute:
            return False
        
        return True
    
    def _send_message(self, content: str, webhook_url: Optional[str] = None, embed: Optional[Dict] = None) -> bool:
        """Send a message to Discord."""
        url = webhook_url or self.config.webhook_url
        if not url:
            logger.debug("No Discord webhook configured")
            return False
        
        with self._lock:
            if not self._can_send():
                # Queue for later
                self._message_queue.append({
                    "content": content,
                    "embed": embed,
                    "url": url,
                    "timestamp": time.time()
                })
                logger.debug("Message queued due to rate limit")
                return False
            
            self._last_send_time = time.time()
            self._send_count_minute += 1
        
        try:
            payload = {}
            if content:
                payload["content"] = content[:2000]  # Discord limit
            if embed:
                payload["embeds"] = [embed]
            
            response = requests.post(
                url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in (200, 204):
                self._daily_stats["alerts_sent"] += 1
                return True
            else:
                logger.warning(f"Discord webhook returned {response.status_code}: {response.text[:200]}")
                return False
                
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")
            return False
    
    def _format_price(self, price: float) -> str:
        """Format price for display."""
        if price >= 1000:
            return f"${price:,.0f}"
        elif price >= 1:
            return f"${price:.2f}"
        else:
            return f"${price:.6f}"
    
    def _format_pct(self, pct: float) -> str:
        """Format percentage for display."""
        return f"{'+' if pct >= 0 else ''}{pct:.2f}%"
    
    def _create_embed(self, 
                      title: str, 
                      description: str, 
                      color: int,
                      fields: Optional[List[Dict]] = None,
                      footer: Optional[str] = None) -> Dict:
        """Create a Discord embed."""
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if fields:
            embed["fields"] = fields
        
        if footer:
            embed["footer"] = {"text": footer}
        
        return embed
    
    # ==================== Signal Notifications ====================
    
    def notify_signal_detected(self,
                               symbol: str,
                               signal: str,  # "BUY" or "SELL"
                               confidence: float,
                               strategy: str,
                               reasons: List[str],
                               price: float) -> bool:
        """Notify when a high-confidence trade signal is detected."""
        if not self.config.notify_signals:
            return False
        
        self._reset_daily_stats()
        self._daily_stats["signals_detected"] += 1
        
        emoji = "🔔" if signal == "BUY" else "🔻"
        color = 0x00FF00 if signal == "BUY" else 0xFF0000  # Green / Red
        
        title = f"{emoji} Signal Detected: {symbol}"
        description = f"**{signal}** signal with {confidence:.0%} confidence"
        
        fields = [
            {"name": "Price", "value": self._format_price(price), "inline": True},
            {"name": "Strategy", "value": strategy, "inline": True},
            {"name": "Confidence", "value": f"{confidence:.0%}", "inline": True}
        ]
        
        if self.config.include_reasoning and reasons:
            reasons_text = "\n".join(f"• {r}" for r in reasons[:5])
            fields.append({"name": "Reasoning", "value": reasons_text, "inline": False})
        
        embed = self._create_embed(title, description, color, fields, "Deploying bot...")
        
        return self._send_message("", embed=embed)
    
    def notify_bot_deployed(self,
                            bot_id: int,
                            symbol: str,
                            strategy: str,
                            entry_price: float,
                            size: float,
                            stop_loss: Optional[float],
                            take_profit: Optional[float],
                            confidence: float,
                            is_paper: bool = True) -> bool:
        """Notify when a bot is deployed and enters a trade."""
        if not self.config.notify_entries:
            return False
        
        self._reset_daily_stats()
        self._daily_stats["trades_opened"] += 1
        
        mode = "Paper" if is_paper else "Live"
        emoji = "📈" if not is_paper else "📊"
        color = 0x00BFFF  # DeepSkyBlue
        
        title = f"{emoji} Bot Deployed - {symbol} ({mode})"
        description = f"Entered **LONG** position using **{strategy}** strategy"
        
        fields = [
            {"name": "Entry Price", "value": self._format_price(entry_price), "inline": True},
            {"name": "Size", "value": f"${size:.2f}", "inline": True},
            {"name": "Confidence", "value": f"{confidence:.0%}", "inline": True}
        ]
        
        if stop_loss:
            sl_pct = (stop_loss - entry_price) / entry_price * 100
            fields.append({"name": "Stop Loss", "value": f"{self._format_price(stop_loss)} ({sl_pct:+.1f}%)", "inline": True})
        
        if take_profit:
            tp_pct = (take_profit - entry_price) / entry_price * 100
            fields.append({"name": "Take Profit", "value": f"{self._format_price(take_profit)} ({tp_pct:+.1f}%)", "inline": True})
        
        fields.append({"name": "Bot ID", "value": f"#{bot_id}", "inline": True})
        
        embed = self._create_embed(title, description, color, fields)
        
        return self._send_message("", embed=embed)
    
    def notify_trade_update(self,
                            bot_id: int,
                            symbol: str,
                            update_type: str,  # "TP_HIT", "STOP_MOVED", "TRAILING_ACTIVATED", etc.
                            details: Dict[str, Any]) -> bool:
        """Notify about trade updates (partial TP, stop moved, etc.)."""
        if not self.config.notify_updates:
            return False
        
        emoji_map = {
            "TP_HIT": "🎯",
            "STOP_MOVED": "🛡️",
            "TRAILING_ACTIVATED": "📐",
            "SAFETY_ORDER": "🔄",
            "BREAKEVEN": "⚖️"
        }
        
        emoji = emoji_map.get(update_type, "📝")
        color = 0xFFD700  # Gold
        
        title = f"{emoji} Trade Update: {symbol}"
        
        messages = {
            "TP_HIT": f"Take Profit {details.get('level', 1)} hit!",
            "STOP_MOVED": f"Stop loss moved to {self._format_price(details.get('new_stop', 0))}",
            "TRAILING_ACTIVATED": f"Trailing stop activated at {self._format_price(details.get('trail_price', 0))}",
            "SAFETY_ORDER": f"Safety order {details.get('order_num', 1)} filled",
            "BREAKEVEN": "Stop moved to breakeven!"
        }
        
        description = messages.get(update_type, f"Update: {update_type}")
        
        fields = []
        if "pnl" in details:
            fields.append({"name": "P/L", "value": self._format_pct(details["pnl"]), "inline": True})
        if "new_avg" in details:
            fields.append({"name": "New Avg", "value": self._format_price(details["new_avg"]), "inline": True})
        if "remaining" in details:
            fields.append({"name": "Remaining", "value": f"{details['remaining']:.0%}", "inline": True})
        
        fields.append({"name": "Bot ID", "value": f"#{bot_id}", "inline": True})
        
        embed = self._create_embed(title, description, color, fields)
        
        return self._send_message("", embed=embed)
    
    def notify_trade_closed(self,
                            bot_id: int,
                            symbol: str,
                            exit_reason: str,  # "TP", "SL", "MANUAL", etc.
                            entry_price: float,
                            exit_price: float,
                            pnl_usd: float,
                            pnl_pct: float,
                            duration_hours: float,
                            strategy: str) -> bool:
        """Notify when a trade is closed."""
        if not self.config.notify_exits:
            return False
        
        self._reset_daily_stats()
        self._daily_stats["trades_closed"] += 1
        self._daily_stats["total_pnl"] += pnl_usd
        if pnl_usd >= 0:
            self._daily_stats["winners"] += 1
        else:
            self._daily_stats["losers"] += 1
        
        is_win = pnl_usd >= 0
        emoji = "💰" if is_win else "📉"
        color = 0x00FF00 if is_win else 0xFF0000
        
        title = f"{emoji} Trade Closed: {symbol}"
        description = f"Exited via **{exit_reason}** | {self._format_pct(pnl_pct)} ({'+' if pnl_usd >= 0 else ''}${pnl_usd:.2f})"
        
        fields = [
            {"name": "Entry", "value": self._format_price(entry_price), "inline": True},
            {"name": "Exit", "value": self._format_price(exit_price), "inline": True},
            {"name": "Duration", "value": f"{duration_hours:.1f}h", "inline": True},
            {"name": "Strategy", "value": strategy, "inline": True},
            {"name": "Bot ID", "value": f"#{bot_id}", "inline": True}
        ]
        
        embed = self._create_embed(title, description, color, fields)
        
        return self._send_message("", embed=embed)
    
    def notify_strategy_switch(self,
                               bot_id: int,
                               symbol: str,
                               old_strategy: str,
                               new_strategy: str,
                               reason: str) -> bool:
        """Notify when strategy is switched mid-trade."""
        if not self.config.notify_strategy_changes:
            return False
        
        emoji = "🔄"
        color = 0x9932CC  # DarkOrchid
        
        title = f"{emoji} Strategy Change: {symbol}"
        description = f"**{old_strategy}** → **{new_strategy}**"
        
        fields = [
            {"name": "Reason", "value": reason, "inline": False},
            {"name": "Bot ID", "value": f"#{bot_id}", "inline": True}
        ]
        
        embed = self._create_embed(title, description, color, fields)
        
        return self._send_message("", embed=embed)
    
    def notify_risk_alert(self,
                          alert_type: str,  # "DRAWDOWN", "VAR_EXCEEDED", "CORRELATION", etc.
                          severity: str,  # "warning", "critical"
                          message: str,
                          details: Dict[str, Any]) -> bool:
        """Notify about risk management alerts."""
        if not self.config.notify_risk_alerts:
            return False
        
        emoji = "⚠️" if severity == "warning" else "🚨"
        color = 0xFFA500 if severity == "warning" else 0xFF0000  # Orange / Red
        
        title = f"{emoji} Risk Alert: {alert_type}"
        description = message
        
        fields = []
        for key, value in details.items():
            if isinstance(value, float):
                if "pct" in key.lower():
                    value = self._format_pct(value * 100)
                else:
                    value = f"{value:.2f}"
            fields.append({"name": key.replace("_", " ").title(), "value": str(value), "inline": True})
        
        embed = self._create_embed(title, description, color, fields)
        
        return self._send_message("", embed=embed)
    
    def notify_daily_summary(self,
                             date: str,
                             total_equity: float,
                             daily_pnl: float,
                             daily_pnl_pct: float,
                             active_bots: int,
                             trades_closed: int,
                             winners: int,
                             losers: int,
                             sharpe_ratio: float,
                             var_95: float,
                             drawdown: float) -> bool:
        """Send daily performance summary."""
        if not self.config.notify_daily_summary:
            return False
        
        is_positive = daily_pnl >= 0
        emoji = "📊" if is_positive else "📉"
        color = 0x00FF00 if is_positive else 0xFF6347  # Green / Tomato
        
        win_rate = winners / trades_closed * 100 if trades_closed > 0 else 0
        
        title = f"{emoji} Daily Summary - {date}"
        description = f"P/L: **{'+' if daily_pnl >= 0 else ''}${daily_pnl:.2f}** ({self._format_pct(daily_pnl_pct)})"
        
        # Performance emoji
        if daily_pnl_pct > 2:
            perf_emoji = "🚀"
        elif daily_pnl_pct > 0:
            perf_emoji = "✅"
        elif daily_pnl_pct > -1:
            perf_emoji = "😐"
        else:
            perf_emoji = "💔"
        
        description += f" {perf_emoji}"
        
        fields = [
            {"name": "Equity", "value": f"${total_equity:,.2f}", "inline": True},
            {"name": "Active Bots", "value": str(active_bots), "inline": True},
            {"name": "Trades Closed", "value": str(trades_closed), "inline": True},
            {"name": "Win Rate", "value": f"{win_rate:.0f}% ({winners}W/{losers}L)", "inline": True},
            {"name": "Sharpe Ratio", "value": f"{sharpe_ratio:.2f}", "inline": True},
            {"name": "95% VaR", "value": f"${var_95:.2f}", "inline": True},
            {"name": "Drawdown", "value": self._format_pct(drawdown), "inline": True}
        ]
        
        # Sharpe emoji
        if sharpe_ratio >= 2:
            sharpe_comment = "🌟 Excellent!"
        elif sharpe_ratio >= 1:
            sharpe_comment = "👍 Good"
        elif sharpe_ratio >= 0:
            sharpe_comment = "📈 Acceptable"
        else:
            sharpe_comment = "⚠️ Review needed"
        
        fields.append({"name": "Risk-Adjusted", "value": sharpe_comment, "inline": True})
        
        footer = "Good trading day!" if is_positive else "Tomorrow's another day. Stay disciplined!"
        
        embed = self._create_embed(title, description, color, fields, footer)
        
        # Use status webhook if available
        webhook = self.config.status_webhook_url or self.config.webhook_url
        return self._send_message("", webhook_url=webhook, embed=embed)
    
    def notify_error(self,
                     error_type: str,
                     message: str,
                     bot_id: Optional[int] = None,
                     symbol: Optional[str] = None) -> bool:
        """Notify about errors."""
        if not self.config.notify_errors:
            return False
        
        emoji = "❌"
        color = 0xFF0000
        
        title = f"{emoji} Error: {error_type}"
        description = message[:500]  # Truncate long messages
        
        fields = []
        if bot_id:
            fields.append({"name": "Bot ID", "value": f"#{bot_id}", "inline": True})
        if symbol:
            fields.append({"name": "Symbol", "value": symbol, "inline": True})
        
        embed = self._create_embed(title, description, color, fields)
        
        return self._send_message("", embed=embed)
    
    # ==================== Simple Text Notifications ====================
    
    def send_simple(self, message: str) -> bool:
        """Send a simple text message."""
        return self._send_message(message)
    
    def send_to_status(self, message: str) -> bool:
        """Send to status channel."""
        webhook = self.config.status_webhook_url or self.config.webhook_url
        return self._send_message(message, webhook_url=webhook)
    
    # ==================== Utility Methods ====================
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get current daily stats."""
        self._reset_daily_stats()
        return self._daily_stats.copy()
    
    def process_queue(self) -> int:
        """Process queued messages. Returns number processed."""
        processed = 0
        
        while self._message_queue and self._can_send():
            msg = self._message_queue.popleft()
            
            # Skip old messages (>5 min)
            if time.time() - msg["timestamp"] > 300:
                continue
            
            if self._send_message(msg["content"], msg.get("url"), msg.get("embed")):
                processed += 1
            
            time.sleep(0.5)  # Small delay between queued messages
        
        return processed


# Singleton instance
_discord_notifier: Optional[DiscordNotifier] = None


def get_discord_notifier() -> DiscordNotifier:
    """Get or create the Discord notifier singleton."""
    global _discord_notifier
    if _discord_notifier is None:
        _discord_notifier = DiscordNotifier()
    return _discord_notifier


def configure_discord(webhook_url: str, status_webhook_url: str = "") -> DiscordNotifier:
    """Configure and return the Discord notifier."""
    global _discord_notifier
    config = NotificationConfig(
        webhook_url=webhook_url,
        status_webhook_url=status_webhook_url
    )
    _discord_notifier = DiscordNotifier(config)
    return _discord_notifier
