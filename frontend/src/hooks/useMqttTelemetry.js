import { useEffect, useRef, useState } from "react";
import mqtt from "mqtt";

export default function useMqttTelemetry() {
  const clientRef = useRef(null);
  const [status, setStatus] = useState("disconnected");
  const [deviceId, setDeviceId] = useState("");
  const [telemetry, setTelemetry] = useState(null);
  const [inference, setInference] = useState(null);
  const [lastMessageAt, setLastMessageAt] = useState(null);
  const [error, setError] = useState("");

  const disconnect = () => {
    const client = clientRef.current;
    clientRef.current = null;
    if (client) client.end(true);
    setStatus("disconnected");
    setDeviceId("");
  };

  const connect = ({ brokerUrl, deviceId: requestedDeviceId }) => {
    const normalizedDeviceId = requestedDeviceId.trim();
    let broker;

    try {
      broker = new URL(brokerUrl.trim());
    } catch {
      setError("Enter a valid broker WebSocket URL, such as ws://localhost:9001/mqtt.");
      return;
    }

    if (!["ws:", "wss:"].includes(broker.protocol)) {
      setError("The browser connection must use ws:// or wss://.");
      return;
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(normalizedDeviceId)) {
      setError("Device ID can contain letters, numbers, hyphens, and underscores.");
      return;
    }

    if (clientRef.current) clientRef.current.end(true);
    setTelemetry(null);
    setInference(null);
    setLastMessageAt(null);
    setError("");
    setDeviceId(normalizedDeviceId);
    setStatus("connecting");

    const client = mqtt.connect(broker.toString(), {
      clientId: `adaptive-web-${Math.random().toString(16).slice(2, 10)}`,
      clean: true,
      connectTimeout: 8000,
      keepalive: 30,
      reconnectPeriod: 3000,
    });
    clientRef.current = client;

    client.on("connect", () => {
      if (clientRef.current !== client) return;
      setStatus("connected");
      client.subscribe([
        `esp32/${normalizedDeviceId}/telemetry`,
        `esp32/${normalizedDeviceId}/inference`,
      ], (subscribeError) => {
        if (subscribeError) setError(`Could not subscribe: ${subscribeError.message}`);
      });
    });
    client.on("reconnect", () => {
      if (clientRef.current === client) setStatus("reconnecting");
    });
    client.on("offline", () => {
      if (clientRef.current === client) setStatus("reconnecting");
    });
    client.on("close", () => {
      if (clientRef.current === client) setStatus("disconnected");
    });
    client.on("error", (connectionError) => {
      if (clientRef.current === client) {
        setStatus("error");
        setError(connectionError.message || "MQTT connection failed. Check the broker and WebSocket port.");
      }
    });
    client.on("message", (topic, payload) => {
      if (clientRef.current !== client) return;
      try {
        const message = JSON.parse(payload.toString());
        if (topic.endsWith("/telemetry")) {
          setTelemetry(message);
          if (["physical", "simulated"].includes(message.source)) {
            const timestamp = typeof message.timestamp === "string" ? message.timestamp : new Date().toISOString();
            fetch("/api/telemetry", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                node_id: message.device_id || normalizedDeviceId,
                timestamp,
                source: message.source,
                voltage_v: message.voltage_v ?? message.bus_voltage_v,
                current_a: message.current_a ?? (message.shunt_current_ma == null ? null : Number(message.shunt_current_ma) / 1000),
                power_w: message.power_w,
                energy_j: message.energy_j,
                temperature_ambient_c: message.temperature_c,
                humidity_pct: message.humidity_pct,
                battery_pct: message.battery_pct,
                hardware_status: message.hardware_status || "degraded",
                quality: message.quality || "partial",
              }),
            }).catch(() => {});
          }
        }
        if (topic.endsWith("/inference")) setInference(message);
        setLastMessageAt(Date.now());
        setError("");
      } catch {
        setError(`Received invalid JSON on ${topic}.`);
      }
    });
  };

  useEffect(() => () => {
    if (clientRef.current) clientRef.current.end(true);
  }, []);

  return { status, deviceId, telemetry, inference, lastMessageAt, error, connect, disconnect };
}
