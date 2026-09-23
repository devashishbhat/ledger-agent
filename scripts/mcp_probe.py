"""Talk to the MCP server directly, bypassing the Inspector, with timings."""
import json
import subprocess
import sys
import time

proc = subprocess.Popen(
    ["uv", "run", "python", "-m", "ledger.mcp_server"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1,
)

def send(obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()

def read(label):
    start = time.time()
    line = proc.stdout.readline()
    print(f"{label}: {time.time() - start:.1f}s -> {line[:200] if line else 'NOTHING'}")
    return line

send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                 "clientInfo": {"name": "probe", "version": "1"}}})
read("initialize")
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
read("tools/list")
send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
      "params": {"name": "list_filings", "arguments": {}}})
read("list_filings")

proc.kill()
print("\n--- stderr ---")
print(proc.stderr.read()[:2000])