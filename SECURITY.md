# Security Policy — RoadSense AI

RoadSense AI is designed with a defense-in-depth security model to safeguard municipal road infrastructure records, authentication sessions, and geospatial metadata.

---

## Supported Versions

| Version | Supported          | Security Maintenance Status |
| ------- | ------------------ | --------------------------- |
| 2.0.x   | :white_check_mark: | Active production support   |
| < 2.0   | :x:                | Deprecated                  |

---

## Reporting a Vulnerability

If you discover a security vulnerability, authentication flaw, or potential sensitive data exposure within RoadSense AI, please report it privately:

1. **GitHub Private Vulnerability Reporting:**
   Use GitHub's [Private Vulnerability Reporting](https://github.com/itsananyaaa/roadsense-ai/security/advisories/new) interface on the repository.
2. **Direct Security Contact:**
   Contact the repository maintainer directly via GitHub Security Advisory.

Please include:
- A description of the vulnerability and its potential impact.
- Step-by-step reproduction instructions or proof-of-concept payload.
- Affected endpoints, modules, or configurations.

**Please do NOT file public GitHub Issues for undisclosed security vulnerabilities.**

---

## Security Design & Defaults

1. **Fail-Closed Authentication:**
   The backend reads credentials from environment variables (`ROADSENSE_ADMIN_USER` and `ROADSENSE_ADMIN_PASS`). If these are not configured, all protected data endpoints fail closed and return `HTTP 401 Unauthorized`.
2. **Argon2id Password Hashing:**
   Operator passwords are never stored in plaintext and never logged. Password hashing utilizes memory-hard `argon2id` with constant-time verification against timing attacks.
3. **Session Integrity:**
   HTTP sessions utilize HMAC-SHA256 cryptographically signed cookies (`roadsense_session`) marked `HttpOnly` and `SameSite=Lax`.
4. **Brute-Force Protection:**
   A sliding-window per-IP rate limiter restricts failed login attempts to 5 attempts per 60-second window before locking out for the duration (`HTTP 429 Too Many Requests`).
5. **CSV Formula Injection Mitigation:**
   All exported CSV data prefixes potential execution characters (`=`, `+`, `-`, `@`) with an apostrophe (`'`) and encodes files with `utf-8-sig`.
6. **Strict Location Provenance:**
   The platform never hallucinates or interpolates GPS coordinates. Unlocated distress reports remain explicitly marked as `UNKNOWN` and are excluded from geospatial dispatch maps.
