import os
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Iterable

from textclassification_app.utils import print_title


def send_mail(to: Iterable, subject, body, attachment_dir=None):
    sender_email = "projectlev2021@gmail.com"
    password = "project_lev_2021"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ", ".join(to)
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    folder = os.path.join(Path(__file__).parent.parent.parent, "results", "excel")

    if attachment_dir:
        for file in os.listdir(folder):
            filename = os.path.join(folder, file)
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(open(filename, 'rb').read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"' % file)
            message.attach(part)

    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, to, text)


def send_results_by_email(receiver_email):
    print_title("Sends results by email")
    subject = "The classification results are ready"
    body = "The run is complete and the classification results are attached to this email.\n" \
           "Also, the run results are in the results folder on the server.\n" \
           "-- DO NOT FORGET TO TURN OFF THE SERVER AFTER USE --"
    folder = os.path.join(Path(__file__).parent.parent.parent, "results", "excel")
    send_mail(receiver_email, subject, body, folder)


if __name__ == '__main__':
    receiver_email = ['natanmanor@gmail.com']
    send_mail(receiver_email, "test", "It's a test")
