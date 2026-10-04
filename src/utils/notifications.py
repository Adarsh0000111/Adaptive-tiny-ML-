import logging
import requests
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AlertDispatcher:
    """
    Real-Time Security & Battery Alert Dispatcher supporting Discord and Telegram Webhooks.
    """
    def __init__(self, discord_webhook_url=None, telegram_bot_token=None, telegram_chat_id=None):
        self.discord_webhook_url = discord_webhook_url
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id

    def send_discord_alert(self, title, description, color=15158332, fields=None):
        """
        Send formatted embed alert to Discord Webhook.
        """
        if not self.discord_webhook_url:
            logger.debug("Discord webhook URL not configured. Skipping Discord alert.")
            return False

        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color, # Default Red: 15158332, Yellow: 16776960, Green: 3066993
                "fields": fields or [],
                "footer": {"text": "Adaptive TinyML IIoT IDS Alert System"}
            }]
        }

        try:
            response = requests.post(self.discord_webhook_url, json=payload, timeout=5)
            if response.status_code in [200, 204]:
                logger.info(f"Discord alert successfully sent: {title}")
                return True
            else:
                logger.error(f"Failed to send Discord alert: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error dispatching Discord alert: {e}")
            return False

    def send_telegram_alert(self, message):
        """
        Send text alert to Telegram Chat.
        """
        if not (self.telegram_bot_token and self.telegram_chat_id):
            logger.debug("Telegram credentials not configured. Skipping Telegram alert.")
            return False

        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.info("Telegram alert successfully sent.")
                return True
            else:
                logger.error(f"Failed to send Telegram alert: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error dispatching Telegram alert: {e}")
            return False

    def dispatch_threat_alert(self, threat_type, severity_score, model_used, confidence, battery_pct):
        """
        Unified dispatch for security threats.
        """
        title = f"🚨 HIGH SEVERITY IIOT INTRUSION DETECTED: {threat_type}"
        description = f"An anomaly matching `{threat_type}` was identified by the adaptive TinyML engine."
        fields = [
            {"name": "Threat Severity", "value": f"{severity_score}/100", "inline": True},
            {"name": "Selected Model Tier", "value": model_used, "inline": True},
            {"name": "Model Confidence", "value": f"{confidence * 100:.1f}%", "inline": True},
            {"name": "Edge Battery Level", "value": f"{battery_pct:.1f}%", "inline": True}
        ]
        
        self.send_discord_alert(title, description, color=15158332, fields=fields)
        
        tg_msg = f"🚨 *IIoT Threat Alert: {threat_type}*\n" \
                 f"• Severity: {severity_score}/100\n" \
                 f"• Model: `{model_used}`\n" \
                 f"• Confidence: {confidence*100:.1f}%\n" \
                 f"• Battery: {battery_pct:.1f}%"
        self.send_telegram_alert(tg_msg)
