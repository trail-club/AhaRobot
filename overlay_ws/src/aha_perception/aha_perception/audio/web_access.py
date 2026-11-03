"""Which requests the debug UI answers. No ROS or aiohttp dependency."""

from __future__ import annotations

LOOPBACK = {"127.0.0.1", "localhost", "[::1]"}


def allowed(host: str, origin: str | None, bind_host: str) -> bool:
    """False for other sites: browsers do not apply the same-origin policy to
    WebSockets, and DNS rebinding reaches a port bound to the loopback address.

    `host` and `origin` are the request's headers. A client without an Origin is not
    a page, so it is local software already.
    """
    name = host.rpartition(":")[0] or host
    if bind_host in LOOPBACK and name not in LOOPBACK:
        return False
    return not origin or origin in (f"http://{host}", f"https://{host}")
