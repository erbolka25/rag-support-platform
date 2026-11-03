import json
import re

import pytest

from support_copilot.api.app import STATIC_DIR
from support_copilot.drafting import Reason
from support_copilot.models import TicketStatus

STATIC_FILES = ["app.css", "app.js", "common.js", "i18n.json", "favicon.svg"]


def flat_keys(tree, prefix=""):
    keys = set()
    for key, value in tree.items():
        path = f"{prefix}{key}"
        keys |= flat_keys(value, f"{path}.") if isinstance(value, dict) else {path}
    return keys


def test_index_serves_operator_ui(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert resp.headers["cache-control"] == "no-store"
    for name in ["app.css", "common.js", "app.js"]:
        assert f"/static/{name}" in resp.text


@pytest.mark.parametrize("name", STATIC_FILES)
def test_static_files_are_served(client, name):
    resp = client.get(f"/static/{name}")
    assert resp.status_code == 200
    assert resp.content == (STATIC_DIR / name).read_bytes()
    # Revalidated on every load, so an updated app.js is never stale.
    assert resp.headers["cache-control"] == "no-cache"


def test_swagger_and_api_are_unchanged(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/" not in paths  # the page is not part of the API schema
    assert {"/health", "/kb", "/kb/upload", "/tickets"} <= set(paths)


def test_translations_have_the_same_keys():
    catalog = json.loads((STATIC_DIR / "i18n.json").read_text(encoding="utf-8"))
    assert set(catalog) == {"ru", "en"}
    assert flat_keys(catalog["ru"]) == flat_keys(catalog["en"])


def test_ui_keys_used_in_scripts_exist():
    catalog = json.loads((STATIC_DIR / "i18n.json").read_text(encoding="utf-8"))
    known = flat_keys(catalog["ru"])
    sources = "".join(
        (STATIC_DIR / name).read_text(encoding="utf-8") for name in ["app.js", "index.html"]
    )
    used = set(re.findall(r"""["'`](ui(?:\.[a-z_]+)+)["'`]""", sources))
    assert used, "no i18n keys found"
    assert used <= known, used - known


def test_every_ticket_status_has_labels():
    # app.js builds these keys from the status value: ui.tickets.status_<s> and filter_<s>.
    catalog = json.loads((STATIC_DIR / "i18n.json").read_text(encoding="utf-8"))
    known = flat_keys(catalog["ru"])
    for status in TicketStatus:
        assert f"ui.tickets.status_{status}" in known
        assert f"ui.tickets.filter_{status}" in known


def test_every_escalation_reason_has_human_text():
    # The card shows ui.reasons.<code> instead of the raw code; app.js lists the known codes.
    catalog = json.loads((STATIC_DIR / "i18n.json").read_text(encoding="utf-8"))
    codes = {v for k, v in vars(Reason).items() if k.isupper()}
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    for lang in ("ru", "en"):
        assert codes | {"unknown"} <= set(catalog[lang]["ui"]["reasons"])
    for code in codes:
        assert f'"{code}"' in app_js, code


def test_scripts_never_build_html_from_strings():
    # User and server text must go through textContent / el(), never through the HTML parser.
    forbidden = re.compile(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(")
    for name in ["app.js", "common.js"]:
        source = (STATIC_DIR / name).read_text(encoding="utf-8")
        assert not forbidden.search(source), name
