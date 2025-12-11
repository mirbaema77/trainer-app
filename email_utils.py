import os
import mailchimp_transactional as MailchimpTransactional
from mailchimp_transactional.api_client import ApiClientError

MANDRILL_API_KEY = os.getenv("MANDRILL_API_KEY")
MAIL_FROM_ADDRESS = os.getenv("MAIL_FROM_ADDRESS")

def send_email(to_email, subject, html_content):
    """
    Sends an email using Mailchimp Transactional (Mandrill)
    """

    if not MANDRILL_API_KEY:
        print("ERROR: MANDRILL_API_KEY is missing")
        return False

    try:
        client = MailchimpTransactional.Client(MANDRILL_API_KEY)

        message = {
            "from_email": MAIL_FROM_ADDRESS,
            "subject": subject,
            "html": html_content,
            "to": [{"email": to_email, "type": "to"}],
        }

        result = client.messages.send({"message": message})
        print("Email sent:", result)
        return True

    except ApiClientError as error:
        print("Mandrill API error:", error.text)
        return False
