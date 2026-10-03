import * as vscode from "vscode";
import * as crypto from "crypto";
import * as net from "net";

const HOST = "127.0.0.1";
const PORT = 9761;

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.commands.registerCommand("artrtic.openPreview", () => openPreview(context))
  );
}

export function deactivate() {}

function openPreview(context: vscode.ExtensionContext) {
  const panel = vscode.window.createWebviewPanel(
    "artrticPreview",
    "ArtRTic",
    vscode.ViewColumn.Beside,
    { enableScripts: true }
  );
  panel.webview.html = html();

  const socket = net.connect(PORT, HOST);
  let buffered = Buffer.alloc(0);
  let upgraded = false;

  socket.on("connect", () => {
    const key = crypto.randomBytes(16).toString("base64");
    socket.write(
      `GET / HTTP/1.1\r\nHost: ${HOST}:${PORT}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: ${key}\r\nSec-WebSocket-Version: 13\r\n\r\n`
    );
  });

  socket.on("data", (chunk) => {
    buffered = Buffer.concat([buffered, chunk]);
    if (!upgraded) {
      const headerEnd = buffered.indexOf("\r\n\r\n");
      if (headerEnd < 0) {
        return;
      }
      upgraded = true;
      buffered = buffered.subarray(headerEnd + 4);
    }
    while (buffered.length >= 2) {
      const length = buffered[1] & 0x7f;
      let header = 2;
      let payloadLength = length;
      if (length === 126) {
        if (buffered.length < 4) return;
        payloadLength = buffered.readUInt16BE(2);
        header = 4;
      } else if (length === 127) {
        if (buffered.length < 10) return;
        payloadLength = Number(buffered.readBigUInt64BE(2));
        header = 10;
      }
      if (buffered.length < header + payloadLength) return;
      const opcode = buffered[0] & 0x0f;
      const payload = buffered.subarray(header, header + payloadLength);
      buffered = buffered.subarray(header + payloadLength);
      if (opcode === 0x2) {
        panel.webview.postMessage({
          type: "jpeg",
          data: Buffer.from(payload).toString("base64"),
        });
      }
    }
  });

  panel.webview.onDidReceiveMessage((message) => {
    if (message?.type !== "look") return;
    const text = `look ${message.dx} ${message.dy} ${message.forward} ${message.strafe}`;
    const body = Buffer.from(text);
    const mask = crypto.randomBytes(4);
    const frame = Buffer.alloc(6 + body.length);
    frame[0] = 0x81;
    frame[1] = 0x80 | body.length;
    mask.copy(frame, 2);
    for (let i = 0; i < body.length; i++) {
      frame[6 + i] = body[i] ^ mask[i % 4];
    }
    socket.write(frame);
  });

  panel.onDidDispose(() => socket.destroy());
  context.subscriptions.push({ dispose: () => socket.destroy() });
}

function html(): string {
  return `<!DOCTYPE html>
<html><body style="margin:0;background:#111;color:#ddd;font-family:sans-serif">
<img id="frame" style="width:100%;height:auto;image-rendering:auto" />
<script>
const vscode = acquireVsCodeApi();
let drag = false;
let last = null;
window.addEventListener("message", (event) => {
  if (event.data.type === "jpeg") {
    document.getElementById("frame").src = "data:image/jpeg;base64," + event.data.data;
  }
});
const img = document.getElementById("frame");
img.addEventListener("pointerdown", (event) => { drag = true; last = event; });
window.addEventListener("pointerup", () => { drag = false; last = null; });
window.addEventListener("pointermove", (event) => {
  if (!drag || !last) return;
  vscode.postMessage({ type: "look", dx: (event.clientX - last.clientX) * 0.005, dy: (last.clientY - event.clientY) * 0.005, forward: 0, strafe: 0 });
  last = event;
});
window.addEventListener("keydown", (event) => {
  const step = event.shiftKey ? 40 : 8;
  const key = event.key.toLowerCase();
  const forward = key === "w" ? step : key === "s" ? -step : 0;
  const strafe = key === "d" ? step : key === "a" ? -step : 0;
  if (forward || strafe) vscode.postMessage({ type: "look", dx: 0, dy: 0, forward, strafe });
});
</script>
<p style="padding:8px">Start the engine with ART_RTIC_PREVIEW=1. Drag to look, WASD to move.</p>
</body></html>`;
}
