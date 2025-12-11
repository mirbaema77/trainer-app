import os
import mailchimp_transactional as MailchimpTransactional
from mailchimp_transactional.api_client import ApiClientError

# Diese Variablen müssen in Railway / deiner .env gesetzt sein:
# MANDRILL_API_KEY  -> dein API-Key von Mailchimp Transactional
# MAIL_FROM_ADDRESS -> z.B. "info@noqe.ch"

MANDRILL_API_KEY = os.environ.get("MANDRILL_API_KEY")
MAIL_FROM_ADDRESS = os.environ.get("MAIL_FROM_ADDRESS", "info@noqe.ch")

MANDRILL_API_KEY = os.environ.get("MANDRILL_API_KEY")
MAIL_FROM_ADDRESS = os.environ.get("MAIL_FROM_ADDRESS", "info@noqe.ch")


_client = None


def _get_client():
    """Gibt einen initialisierten Mandrill-Client zurück oder None, wenn kein API-Key gesetzt ist."""
    global _client
    if _client is None and MANDRILL_API_KEY:
        _client = MailchimpTransactional.Client(MANDRILL_API_KEY)
    return _client


def send_email(to_address: str, subject: str, body_text: str, body_html: str | None = None) -> bool:
    """
    Einfache E-Mail über Mandrill senden (ohne Template).
    Wird von /test-email und vom Passwort-Reset verwendet.
    """
    client = _get_client()
    if not client:
        print("Mandrill client not configured (MANDRILL_API_KEY missing)")
        return False

    if not MAIL_FROM_ADDRESS:
        print("MAIL_FROM_ADDRESS not set")
        return False

    if not to_address:
        print("No recipient address")
        return False

    message = {
        "from_email": MAIL_FROM_ADDRESS,
        "subject": subject,
        "text": body_text or "",
        "html": body_html or body_text or "",
        "to": [{"email": to_address, "type": "to"}],
    }

    try:
        result = client.messages.send({"message": message})
        print("Email sent:", result)
        return True
    except ApiClientError as e:
        print("Mandrill API error in send_email:", e.text)
        return False


def send_email_template(
    to_address: str,
    template_name: str,
    merge_vars: dict | None = None,
    subject: str | None = None,
) -> bool:
    """
    E-Mail über ein Mandrill-Template senden.
    'template_name' muss in Mandrill existieren (z.B. 'registration_welcome').
    merge_vars: {"name": "Max", "email": "..."} etc.
    """
    client = _get_client()
    if not client:
        print("Mandrill client not configured (MANDRILL_API_KEY missing)")
        return False

    if not MAIL_FROM_ADDRESS:
        print("MAIL_FROM_ADDRESS not set")
        return False

    if not to_address:
        print("No recipient address")
        return False

    if merge_vars is None:
        merge_vars = {}

    message = {
        "from_email": MAIL_FROM_ADDRESS,
        "to": [{"email": to_address, "type": "to"}],
        "merge_language": "handlebars",
        "global_merge_vars": [
            {"name": key, "content": value}
            for key, value in merge_vars.items()
        ],
    }

    if subject:
        message["subject"] = subject

    try:
        result = client.messages.send_template(
            {
                "template_name": template_name,
                "template_content": [],
                "message": message,
            }
        )
        print("Template email sent:", result)
        return True
    except ApiClientError as e:
        print("Mandrill API error in send_email_template:", e.text)
        return False
