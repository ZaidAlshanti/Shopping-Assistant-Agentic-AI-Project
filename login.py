"""
Customer sign-up / sign-in against Data/Customers.json.

Passwords are stored as salted hashes (stdlib only, no new packages).
signin() and signup() return the customer record instead of a message,
so the app can keep the customer's ID for the session.
"""

import hashlib
import hmac
import json
import os

CUSTOMERS_FILE = "Data/Customers.json"


def _load() -> list[dict]:
    try:
        with open(CUSTOMERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _save(customers: list[dict]) -> None:
    with open(CUSTOMERS_FILE, "w", encoding="utf-8") as f:
        json.dump(customers, f, indent=4, ensure_ascii=False)


def _hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{salt.hex()}${digest.hex()}"


def _verify(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$")
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 200_000
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


def signup(name: str, email: str, password: str) -> tuple[dict | None, str]:
    """Create an account. Returns (customer, message); customer is None on failure."""
    name, email = name.strip(), email.strip().lower()
    if not name or not email or len(password) < 8:
        return None, "Name and email are required, and the password needs at least 8 characters."

    customers = _load()
    if any(c.get("email") == email for c in customers):
        return None, "Email already exists."

    customer = {
        "id": f"C{len(customers) + 1:03}",
        "name": name,
        "email": email,
        "password_hash": _hash(password),
    }
    customers.append(customer)
    _save(customers)
    return {"id": customer["id"], "name": name}, "Account created successfully!"


def signin(email: str, password: str) -> tuple[dict | None, str]:
    """Check the login. Returns (customer, message); customer is None on failure."""
    email = email.strip().lower()
    customers = _load()
    for c in customers:
        if c.get("email", "").lower() != email:
            continue

        if c.get("password_hash"):
            ok = _verify(password, c["password_hash"])
        elif "password" in c:
            # old plain-text account: accept it once, then store a hash instead
            ok = hmac.compare_digest(str(c["password"]), password)
            if ok:
                c["password_hash"] = _hash(password)
                del c["password"]
                _save(customers)
        else:
            ok = False

        if ok:
            return {"id": c["id"], "name": c["name"]}, f"Welcome back {c['name']}!"
        break
    # same message for a wrong email and a wrong password
    return None, "Invalid email or password."