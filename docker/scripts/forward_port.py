#!/usr/bin/env python3
"""Relay a TCP port to the Docker host network (used by `make forward`).

On Docker Desktop for macOS, `network_mode: host` binds inside the Linux VM, so a
browser on the Mac cannot reach servers in the dev container. This relay runs in
a bridged helper container published on the Mac's localhost and connects to the
same port through the bridge gateway, i.e. the VM's host network.
"""

import asyncio
import socket
import struct
import sys


def gateway():
    with open("/proc/net/route") as routes:
        for line in routes.readlines()[1:]:
            fields = line.split()
            if fields[1] == "00000000":  # default route
                return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    raise SystemExit("No default route")


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except ConnectionError:
        pass
    finally:
        writer.close()


async def main(port):
    target = gateway()

    async def handle(reader, writer):
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                target, port
            )
        except OSError:
            writer.close()
            return
        await asyncio.gather(
            pipe(reader, upstream_writer), pipe(upstream_reader, writer)
        )

    server = await asyncio.start_server(handle, "0.0.0.0", port)
    print(f"Forwarding localhost:{port} -> {target}:{port}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1])))
