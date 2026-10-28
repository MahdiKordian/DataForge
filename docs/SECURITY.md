# Security and privacy

DataForge is meant to run on your own computer or in a private deployment. This page says what it protects against and what it does not.

## What is covered

| Risk | What DataForge does | Where |
|---|---|---|
| Wrong or hostile file | Only `csv, tsv, txt, xlsx, xls, json, parquet`. Size limit (200 MB, change with `DATAFORGE_MAX_MB`). The content must match the extension (real zip for `.xlsx`, `PAR1` for Parquet, no binary in text files). Spreadsheet archives that expand absurdly (zip bombs) are refused. | `security.check_upload` |
| Download from a link (SSRF) | Only `http` and `https`, ports 80/443/8080/8443, no user name in the link. Every address the host resolves to must be public (no loopback, private, link-local, shared 100.64.0.0/10, metadata addresses like 169.254.169.254, IPv4-mapped IPv6). Every redirect is checked again. Size cap while streaming. | `security.safe_fetch` |
| Running code from a file | No pickle anywhere. Saved pipelines are JSON; step types are looked up in a fixed list; unexpected fields are refused. The generated Python script only contains your settings as literals. | `pipeline.from_state`, `persist.to_script` |
| Path traversal in projects | Project names are restricted to a safe pattern and the final path must stay inside the project folder. Writes are atomic. Stored data is checked against a fingerprint on opening. | `security.safe_project_name`, `projects.py` |
| Personal data in outputs | A scan flags columns that look like emails, phone numbers, card numbers (Luhn), IBAN, IP addresses, SSN-style IDs, and columns whose *name* suggests personal data. Example values of flagged columns are hidden in reports and on the Variable types page. Warnings appear before saving a project or a pipeline file that stores learned values. | `security.pii_scan` |
| Personal data in logs | The activity log records step names, counts and settings, never cell values. `security.redact` removes emails and long digit runs from any text that might carry values. | `security.redact` |
| Untrusted pipeline file | Size limit, format and version check, fixed step list. | `persist.loads` |

## What is not covered

- **No login, no per-user separation.** Anyone who can open the app can open every project in the project folder. Do not expose it to the internet without authentication in front of it (a reverse proxy with login, or your platform's access control).
- **The personal-data scan is a hint.** It misses names, addresses and free text unless the column name says so. It can also flag harmless columns.
- **DNS rebinding.** The address is checked when the link is validated and again on each redirect, but a hostile DNS server could answer differently at connection time. For a hostile environment, put the app behind an egress firewall.
- **Memory.** Files are read into memory. The size limit protects against accidents, not against many simultaneous large uploads.
- **Pipeline and project files contain learned values** (category names, means). If a column holds personal data and is encoded, those values are in the file. Treat the files like the data.
- **Streamlit itself** (cookies, XSRF, websocket origin) should be configured for your deployment; see Streamlit's deployment documentation.

## Reporting a security problem

Please do not open a public issue for a vulnerability. Use GitHub's private reporting instead:
**Security tab > Report a vulnerability** in the repository (https://github.com/MahdiKordian/DataForge/security/advisories/new).
If that is not available to you, contact the author through the links in the README and share the details only in private.

Include the DataForge version (shown in the sidebar), what you did, and what you saw. Please give a reasonable time to fix the problem before you make it public.
Ordinary bugs and questions belong in the normal issue tracker.

## Settings

| Variable | Meaning | Default |
|---|---|---|
| `DATAFORGE_MAX_MB` | Largest accepted file | 200 |
| `DATAFORGE_HOME` | Folder for projects (`<folder>/projects`) | `~/.dataforge` |
