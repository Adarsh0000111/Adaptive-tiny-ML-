import { useCallback, useEffect, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

export default function useBackendApi() {
  const [status, setStatus] = useState("checking");
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState(null);
  const [usbDevice, setUsbDevice] = useState(null);
  const [usbResult, setUsbResult] = useState(null);
  const [usbRunning, setUsbRunning] = useState(false);
  const [ablationResult, setAblationResult] = useState(null);
  const [networkedFederated, setNetworkedFederated] = useState(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [healthData, modelsData, usbData, flData] = await Promise.all([
        getJson("/api/health"), getJson("/api/models"), getJson("/api/device/usb/status"), getJson("/api/federated/networked/status"),
      ]);
      setHealth(healthData);
      setModels(modelsData);
      setUsbDevice(usbData);
      setNetworkedFederated(flData);
      if (usbData.last_run) setUsbResult(usbData.last_run);
      else setUsbResult(null);
      setStatus("connected");
      setError("");
    } catch (fetchError) {
      setStatus("disconnected");
      setError(fetchError.message || "API unavailable");
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    let socket;
    let retryTimer;
    let closed = false;
    const base = API_BASE || `${window.location.protocol}//${window.location.host}`;
    const wsBase = base.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
    const connect = () => {
      if (closed) return;
      try {
        socket = new WebSocket(`${wsBase}/ws`);
        socket.onmessage = (event) => {
          try {
            const update = JSON.parse(event.data);
            if (update.type === "usb_live_capture" || update.type === "usb_inference_replay") setUsbResult(update.data);
          } catch { /* Ignore malformed websocket messages. */ }
        };
        socket.onclose = () => { if (!closed) retryTimer = window.setTimeout(connect, 5000); };
      } catch { retryTimer = window.setTimeout(connect, 5000); }
    };
    connect();
    return () => { closed = true; window.clearTimeout(retryTimer); socket?.close(); };
  }, []);

  const requestRun = async (endpoint, body, errorMessage, publishUsbResult = true) => {
    setUsbRunning(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `${response.status} ${response.statusText}`);
      await refresh();
      if (publishUsbResult) setUsbResult(result);
      return result;
    } catch (fetchError) {
      setError(fetchError.message || errorMessage);
      return null;
    } finally {
      setUsbRunning(false);
    }
  };

  const runUsbLiveCapture = ({ interfaceName = "", authorized = false } = {}) => requestRun(
    "/api/device/usb/live-capture",
    { authorized, interface: interfaceName.trim() || null, max_packets: 100, timeout_seconds: 5 },
    "Live IDS capture failed",
  );

  const runUsbReplay = () => requestRun(
    "/api/device/usb/replay", { count: 100, sample_start: 0 }, "ESP32 USB replay failed",
  );

  const runSensorAblation = () => requestRun(
    "/api/experiments/sensor-ablation", { count: 100, sample_start: 0 }, "Sensor ablation failed", false,
  ).then((result) => { if (result) setAblationResult(result); return result; });

  return { status, health, models, usbDevice, usbResult, usbRunning, runUsbLiveCapture, runUsbReplay, runSensorAblation, ablationResult, networkedFederated, error, refresh };
}
