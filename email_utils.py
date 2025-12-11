import os
import mailchimp_transactional as MailchimpTransactional
from mailchimp_transactional.api_client import ApiClientError

MANDRILL_API_KEY = os.getenv("MANDRILL_API_KEY")
MAIL_FROM_ADDRESS = os.getenv("MAIL_FROM_ADDRESS")


def send_email(to_address, subject, text_body, html_body=None):
    """
    Send an email via Mailchimp Transactional (Mandrill).
    text_body: plain text version
    html_body: optional HTML version
    """

    if not MANDRILL_API_KEY:
        print("ERROR: MANDRILL_API_KEY is missing")
        return False

    if not MAIL_FROM_ADDRESS:
        print("ERROR: MAIL_FROM_ADDRESS is missing")
        return False

    try:
        client = MailchimpTransactional.Client(MANDRILL_API_KEY)

        message = {
            "from_email": MAIL_FROM_ADDRESS,
            "subject": subject,
            "text": text_body or "",               # plain text
            "html": html_body or text_body or "",  # html (fallback to text)
            "to": [{"email": to_address, "type": "to"}],
        }

        result = client.messages.send({"message": message})
        print("Email sent:", result)
        return True

    except ApiClientError as error:
        print("Mandrill API error:", error.text)
        return False
