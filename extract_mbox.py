#!/usr/bin/env python3
"""extract_mbox.py

Read an mbox file and write out a CSV with common metadata and the plain-text body.

Output columns: message_id, subject, from, to, cc, date, body

Usage:
  python3 scripts/extract_mbox.py --input "path/to/mbox" --output "data/out.csv"
"""
import argparse
import mailbox
import csv
import email
from email.header import decode_header, make_header
import html
import re


def decode_header_value(value):
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def html_to_text(html_content):
    # Very small HTML -> text fallback: strip tags and unescape entities
    # Not a full HTML parser but good for simple email bodies.
    text = re.sub(r"<style.*?>.*?</style>", "", html_content, flags=re.S|re.I)
    text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.S|re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text)


def clean_quoted_text(text):
    """Return text with common quoted-reply blocks removed.

    - Remove lines that start with '>' (common quoted lines).
    - Truncate the body at common reply/forward separators such as
      '-----Original Message-----', lines like 'On ... wrote:',
      forwarded message markers, or timestamps like '2025年5月23日(...):'.
    - Remove trailing signature separated by a line containing only '--'
    This is a conservative, heuristic cleaner intended to improve
    relevance for downstream AI ingestion.
    """
    if not text:
        return ""
    lines = text.splitlines()
    cleaned_lines = []
    # patterns that indicate the start of quoted/previous messages
    quote_start_patterns = [
        re.compile(r"^\s*>+"),
        re.compile(r"^\s*-{2,} ?Original Message ?-{2,}", re.I),
        re.compile(r"^\s*-----Forwarded message-----", re.I),
        re.compile(r"^On .*wrote:$", re.I),
        re.compile(r"^From:\s+.*", re.I),
        re.compile(r"^Sent:\s+.*", re.I),
        re.compile(r"^To:\s+.*", re.I),
        re.compile(r"^Subject:\s+.*", re.I),
        # Japanese date-time style used in replies in this dataset
        re.compile(r"^\d{4}年\d{1,2}月\d{1,2}日"),
        re.compile(r"^\d{4}-\d{2}-\d{2}"),
    ]

    for i, line in enumerate(lines):
        # If line is a pure signature separator, truncate
        if re.match(r"^\s*--\s*$", line):
            break
        # If any of the quote-start patterns match and it's not the very first line,
        # assume the remainder is quoted/forwarded content and stop.
        matched = False
        for p in quote_start_patterns:
            if p.match(line):
                # If this is a '>' quoted line, skip it rather than truncating.
                if p.pattern == r"^\\s*>+":
                    matched = True
                    break
                # Otherwise, treat as start of previous message and truncate.
                if i > 0:
                    matched = True
                    # indicate truncation by setting a flag and break outer loop
                    cleaned_lines.append('')
                    matched = 'truncate'
                    break
        if matched == 'truncate':
            break
        # skip '>' quoted lines
        if re.match(r"^\s*>+", line):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def get_body(message):
    # Prefer text/plain, if missing attempt to extract text from text/html
    body_parts = []
    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            disp = str(part.get('Content-Disposition'))
            if ctype == 'text/plain' and 'attachment' not in disp:
                try:
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        continue
                    charset = part.get_content_charset() or 'utf-8'
                    body_parts.append(payload.decode(charset, errors='ignore'))
                except Exception:
                    continue
        if body_parts:
            return "\n".join(body_parts).strip()

        # Fallback: try text/html
        for part in message.walk():
            if part.get_content_type() == 'text/html':
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or 'utf-8'
                    html_body = payload.decode(charset, errors='ignore')
                    return html_to_text(html_body).strip()
        return ""
    else:
        payload = message.get_payload(decode=True)
        if payload is None:
            return ""
        charset = message.get_content_charset() or 'utf-8'
        try:
            text = payload.decode(charset, errors='ignore')
        except Exception:
            text = payload.decode('utf-8', errors='ignore') if isinstance(payload, bytes) else str(payload)
        if message.get_content_type() == 'text/html':
            return html_to_text(text).strip()
        return text.strip()


def extract(mbox_path, out_csv_path):
    mbox = mailbox.mbox(mbox_path)
    def escape_md(text):
        return text.replace('`', '\`').replace('*', '\*').replace('_', '\_')
    with open(out_csv_path, 'w', encoding='utf-8') as mdfile:
        for i, message in enumerate(mbox):
            try:
                msg_obj = message if isinstance(message, email.message.Message) else email.message_from_string(str(message))
                mid = decode_header_value(msg_obj.get('Message-ID')) or f"mbox-{i}"
                subject = escape_md(decode_header_value(msg_obj.get('Subject')))
                from_ = escape_md(decode_header_value(msg_obj.get('From')))
                to = escape_md(decode_header_value(msg_obj.get('To')))
                cc = escape_md(decode_header_value(msg_obj.get('Cc')))
                date = escape_md(decode_header_value(msg_obj.get('Date')))
                raw_body = get_body(msg_obj)
                cleaned_body = clean_quoted_text(raw_body)
                body = escape_md(cleaned_body)
                mdfile.write(f"## Subject: {subject}\n")
                mdfile.write(f"- Message-ID: {mid}\n")
                mdfile.write(f"- From: {from_}\n")
                mdfile.write(f"- To: {to}\n")
                mdfile.write(f"- CC: {cc}\n")
                mdfile.write(f"- Date: {date}\n")
                mdfile.write(f"---\n{body}\n\n\n")
            except Exception as e:
                mdfile.write(f"## Subject: \n- Message-ID: error-{i}\n- From: \n- To: \n- CC: \n- Date: \n---\n__EXTRACTION_ERROR__ {str(e)}\n\n\n")


def main():
    parser = argparse.ArgumentParser(description='Extract mbox to CSV for RAG preprocessing')
    parser.add_argument('--input', '-i', required=True, help='Path to mbox file')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    args = parser.parse_args()
    extract(args.input, args.output)


if __name__ == '__main__':
    main()
