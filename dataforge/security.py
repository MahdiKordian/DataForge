# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Input safety and privacy helpers: file checks, safe URL download, personal-data scan, safe project names."""
from __future__ import annotations

import io
import ipaddress
import os
import re
import socket
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd

MAX_MB = float(os.environ.get("DATAFORGE_MAX_MB", 200))
ALLOWED_EXT = {"csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"}
ALLOWED_PORTS = {80, 443, 8080, 8443}
ZIP_MAX_RATIO, ZIP_MAX_MB = 200, 1500


class UnsafeInput(ValueError):
    """The file or link was refused. The message is safe to show to the user."""


# ------------------------------------------------------------------------------------------ files
def safe_filename(name: str) -> str:
    name = os.path.basename(str(name).replace("\\", "/"))
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name[:120] or "file"


def check_upload(name: str, raw: bytes, max_mb: float | None = None) -> str:
    """Raises UnsafeInput unless the bytes look like the kind of file the name claims. Returns the extension."""
    limit = (max_mb if max_mb is not None else MAX_MB) * 1024 * 1024
    name = safe_filename(name)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXT:
        raise UnsafeInput(f"Files of type .{ext or '?'} are not accepted. Use: {', '.join(sorted(ALLOWED_EXT))}.")
    if len(raw) > limit:
        raise UnsafeInput(f"The file is {len(raw) / 1048576:.0f} MB; the limit is {limit / 1048576:.0f} MB.")
    if not raw:
        raise UnsafeInput("The file is empty.")
    head = raw[:8192]
    if ext == "xlsx":
        if not raw.startswith(b"PK"):
            raise UnsafeInput("This is not a real .xlsx file.")
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                total = sum(i.file_size for i in z.infolist())
                if total > ZIP_MAX_MB * 1048576 or total > ZIP_MAX_RATIO * max(len(raw), 1):
                    raise UnsafeInput("The spreadsheet expands to an unreasonable size and was refused.")
        except zipfile.BadZipFile:
            raise UnsafeInput("The .xlsx file is damaged.") from None
    elif ext == "xls":
        if not raw.startswith(b"\xd0\xcf\x11\xe0"):
            raise UnsafeInput("This is not a real .xls file.")
    elif ext == "parquet":
        if not (raw.startswith(b"PAR1") and raw.rstrip(b"\x00").endswith(b"PAR1")):
            raise UnsafeInput("This is not a real Parquet file.")
    elif ext == "json":
        if head.lstrip(b"\xef\xbb\xbf \t\r\n")[:1] not in (b"[", b"{"):
            raise UnsafeInput("This does not look like a JSON table.")
    else:
        if b"\x00" in head or raw.startswith((b"PK", b"MZ", b"\x7fELF", b"%PDF")):
            raise UnsafeInput("This looks like a binary file, not text.")
    return ext


# ------------------------------------------------------------------------------------------- URLs
def _is_public(ip: str) -> bool:
    a = ipaddress.ip_address(ip.split("%")[0])           # an IPv6 scope id ("fe80::1%eth0") is not part of the address
    if a.version == 6 and a.ipv4_mapped is not None:
        return _is_public(str(a.ipv4_mapped))
    # is_global is False for private, loopback, link-local, reserved and shared (100.64.0.0/10) ranges
    return a.is_global and not (a.is_multicast or a.is_unspecified)


def check_url(url: str) -> str:
    """Only plain public web addresses are allowed: no file://, no credentials, no private or internal hosts."""
    p = urlparse(url.strip())
    if p.scheme not in ("http", "https"):
        raise UnsafeInput("Only http and https links are allowed.")
    if p.username or p.password:
        raise UnsafeInput("Links with a user name or password are not allowed.")
    if not p.hostname:
        raise UnsafeInput("The link has no host name.")
    port = p.port or (443 if p.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise UnsafeInput(f"Port {port} is not allowed.")
    try:
        infos = socket.getaddrinfo(p.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise UnsafeInput("The host name could not be found.") from None
    for info in infos:
        if not _is_public(info[4][0]):
            raise UnsafeInput("Links to private or internal addresses are not allowed.")
    return url.strip()


def safe_fetch(url: str, max_mb: float | None = None, timeout: int = 20, max_redirects: int = 3) -> tuple[bytes, str]:
    """Downloads a file with a size cap, checking every redirect. Returns (bytes, final file name)."""
    import requests
    limit = (max_mb if max_mb is not None else MAX_MB) * 1024 * 1024
    cur = check_url(url)
    for _ in range(max_redirects + 1):
        with requests.get(cur, stream=True, timeout=timeout, allow_redirects=False, headers={"User-Agent": "DataForge"}) as r:
            if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
                cur = check_url(urljoin(cur, r.headers.get("Location", "")))
                continue
            r.raise_for_status()
            if int(r.headers.get("Content-Length") or 0) > limit:
                raise UnsafeInput(f"The file is larger than {limit / 1048576:.0f} MB.")
            buf, size = io.BytesIO(), 0
            for chunk in r.iter_content(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise UnsafeInput(f"The file is larger than {limit / 1048576:.0f} MB.")
                buf.write(chunk)
            return buf.getvalue(), safe_filename(urlparse(cur).path.rsplit("/", 1)[-1] or "download.csv")
    raise UnsafeInput("Too many redirects.")


# ------------------------------------------------------------------------------- personal data
_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
_PHONE = re.compile(r"^\+?\d[\d\s().-]{7,17}\d$")
_IPV4 = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_DATE = re.compile(r"^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})$")
_IBAN = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$")
_SSN = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_NAME_HINT = re.compile(r"e-?mail|phone|mobile|tel\b|address|ssn|passport|iban|card|birth|dob|first.?name|last.?name|full.?name|surname", re.I)


def _luhn(s: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", s)]
    if not 13 <= len(d) <= 19:
        return False
    tot = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x = x * 2 - 9 if x * 2 > 9 else x * 2
        tot += x
    return tot % 10 == 0


def pii_scan(df: pd.DataFrame, sample: int = 2000, min_share: float = 0.2) -> pd.DataFrame:
    """Columns whose values look like personal data. A hint, not a guarantee: names and addresses are only caught by column name."""
    rows = []
    d = df.sample(min(len(df), sample), random_state=0) if len(df) > sample else df
    for c in df.columns:
        s = d[c].dropna().astype(str).str.strip()
        if s.empty:
            continue
        found = {}
        for kind, test in (("email", lambda v: bool(_EMAIL.match(v))), ("phone number", lambda v: bool(_PHONE.match(v)) and not _IPV4.match(v) and not _DATE.match(v)),
                           ("IP address", lambda v: bool(_IPV4.match(v))), ("IBAN", lambda v: bool(_IBAN.match(v))),
                           ("ID number (SSN style)", lambda v: bool(_SSN.match(v))), ("card number", _luhn)):
            share = float(s.map(test).mean())
            if share >= min_share:
                found[kind] = share
        if _NAME_HINT.search(str(c)) and not found:
            found["column name suggests personal data"] = 1.0
        for kind, share in found.items():
            rows.append({"column": c, "looks like": kind, "share of values %": round(share * 100, 1)})
    return pd.DataFrame(rows, columns=["column", "looks like", "share of values %"])


def pii_columns(df: pd.DataFrame) -> list:
    return list(pii_scan(df)["column"].unique())


_REDACT = [(re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[email]"), (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[number]"),
           (re.compile(r"\+?\d[\d\s().-]{7,17}\d"), "[number]")]


def redact(text: str) -> str:
    """Removes emails and long digit sequences from a string that is about to be logged or shown in a report."""
    for rx, rep in _REDACT:
        text = rx.sub(rep, str(text))
    return text


# -------------------------------------------------------------------------------- project names
_NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,62}$")


def safe_project_name(name: str) -> str:
    name = str(name).strip()
    if not _NAME_OK.match(name) or ".." in name or name.endswith((".", " ")):
        raise UnsafeInput("Use 1 to 63 letters, digits, spaces, dots, dashes or underscores, starting with a letter or digit.")
    return name


def inside(root: Path, p: Path) -> Path:
    """Returns p resolved, and fails if it would leave root (path traversal)."""
    rp, rr = p.resolve(), root.resolve()
    if rr != rp and rr not in rp.parents:
        raise UnsafeInput("That path is outside the project folder.")
    return rp
