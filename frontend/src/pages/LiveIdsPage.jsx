import { useState } from "react";
import { Activity, ArrowRight, Cpu, Network, Radio, Server, ShieldCheck, Usb } from "lucide-react";

const tierLabels = { lightweight: "Low · ESP32", medium: "Medium · laptop", heavy: "High · laptop" };
const pct = (value) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const sensorDecisionSummary = (context) => {
  if (!context || context.electrical_quality === "unavailable") return "sensor context unavailable";
  const v = context.voltage_v ?? context.reported_voltage_v;
  const a = context.current_a ?? context.reported_current_a;
  const p = context.power_w ?? context.reported_power_w;
  const voltage = v == null ? "—" : `${Number(v).toFixed(2)} V`;
  const ma = a == null ? "—" : `${(Number(a) * 1000).toFixed(2)} mA`;
  const w = p == null ? "—" : `${Number(p).toFixed(3)} W`;
  const ambient = context.ambient_context;
  const env = ambient ? `DHT22 ${ambient.temperature_ambient_c ?? "—"}°C/${ambient.humidity_pct ?? "—"}% (context)` : "DHT22 unavailable";
  const reason = context.electrical_quality_reason ? ` · ${context.electrical_quality_reason}` : "";
  return `INA219 ${context.electrical_quality}: ${voltage}, ${ma}, ${w}${reason} · ${env} · physical event ${context.physical_event ? "detected" : "none"} (not threat evidence)`;
};

export default function LiveIdsPage({ backend }) {
  const [interfaceName, setInterfaceName] = useState("");
  const [authorized, setAuthorized] = useState(false);
  const result = backend.usbResult?.mode === "authorized_live_adaptive_packet_inference" ? backend.usbResult : null;
  const device = backend.usbDevice?.ports?.find((port) => port.accessible);
  const tierCounts = result?.selected_tier_counts || {};
  const sensor = result?.device_telemetry || (backend.usbResult?.mode === "saved_test_replay_on_esp32" ? backend.usbResult.device_telemetry : null);
  const sensorsComplete = sensor?.sensor_presence_quality === "complete"
    || (sensor?.sensor_status && Object.values(sensor.sensor_status).length > 0 && Object.values(sensor.sensor_status).every(Boolean));

  return <>
    <div className="live-hero panel">
      <div className="live-hero-copy">
        <p className="eyebrow"><span className={`pulse ${backend.status === "connected" ? "" : "muted-pulse"}`} /> LIVE PACKET IDS · USB EDGE INFERENCE</p>
        <h1>Adaptive intrusion detection</h1>
        <p>Network packet features remain the IDS input. SARA combines their model risk and confidence with host resources; qualified INA219 supply voltage/current can block ESP32 execution or trigger trusted offload. Ambient and motion sensors stay context only.</p>
      </div>
      <div className="live-hero-aside">
        <div className="live-flow" aria-label="Network packets go to SARA, which routes inference to the ESP32 or laptop">
          <div className="flow-node"><Network size={17} /><span>Network<small>packet features</small></span></div>
          <ArrowRight className="flow-arrow" size={16} />
          <div className="flow-node flow-sara"><ShieldCheck size={18} /><span>SARA<small>risk · confidence · load</small></span></div>
          <div className="flow-outputs"><span><Cpu size={14} />ESP32 <b>Low</b></span><span><Server size={14} />Laptop <b>Med / High</b></span></div>
        </div>
        <div className="live-status-stack">
          <span className={`connection-badge ${backend.status === "connected" ? "connected" : "checking"}`}><i />API {backend.status}</span>
          <span className={`connection-badge ${device ? "connected" : "checking"}`}><i />ESP32 {device?.port || "not detected"}</span>
          <span className="connection-badge"><Usb size={13} /> USB serial · no Wi-Fi</span>
        </div>
      </div>
    </div>

    <div className="panel live-capture-panel">
      <div className="section-heading"><div><h2>Monitor a live capture window</h2><p>Up to 100 packets or five seconds. Tier selection and inference happen inside the packet capture callback.</p></div><span className="connection-badge"><Radio size={13} /> Passive capture</span></div>
      <form className="live-capture-form" onSubmit={(event) => { event.preventDefault(); backend.runUsbLiveCapture({ interfaceName, authorized }); }}>
        <label className="form-label">Network interface <input value={interfaceName} onChange={(event) => setInterfaceName(event.target.value)} placeholder="Blank uses the operating system default" /></label>
        <label className="checkbox-label"><input type="checkbox" checked={authorized} onChange={(event) => setAuthorized(event.target.checked)} /> I am authorized to monitor this interface and approve this capture</label>
        <button className="primary-button live-start-button" type="submit" disabled={!authorized || backend.usbRunning || backend.status !== "connected" || !device}>
          <Activity size={16} />{backend.usbRunning ? "Processing arriving packets…" : "Start authorized live IDS window"}
        </button>
      </form>
      <p className="empty-note">Packet payloads and IP addresses are not stored. Live predictions have no ground-truth labels, so the dashboard reports predictions and model switches, not accuracy. USB inference can slow capture; very high packet rates may be missed.</p>
      {backend.error && <p className="form-error" role="status">{backend.error}</p>}
    </div>

    {result && <>
      <div className="section-heading live-results-heading"><div><h2>Latest live window</h2><p>{result.samples} packets processed in {Number(result.capture_window_seconds || 0).toFixed(2)} s · {new Date(result.completed_at_unix * 1000).toLocaleTimeString()}</p></div><span className="connection-badge connected"><i />Adaptive routing active</span></div>
      <div className="tier-grid live-tier-counts">
        {Object.entries(tierLabels).map(([key, label]) => <article className="panel sara-card" key={key}><small>{label}</small><strong>{tierCounts[key] || 0}</strong><p>{key === "lightweight" ? "Inference ran on ESP32-S3" : "Inference ran on laptop"}</p></article>)}
        <article className="panel sara-card"><small>Model switches</small><strong>{result.model_switches?.length || 0}</strong><p>Changes between decisions in this window</p></article>
      </div>
      {result.model_switches?.length > 0 && <div className="panel live-switch-panel"><h3>Automatic tier changes</h3>{result.model_switches.map((item, index) => <div className="live-switch-row" key={`${item.packet_index}-${index}`}><strong>Packet {item.packet_index + 1}</strong><span>{tierLabels[item.from] || item.from} → {tierLabels[item.to] || item.to}</span><small>Risk {pct(item.risk)} · {item.reason}</small></div>)}</div>}
      <div className="panel live-predictions-panel"><div className="section-heading"><div><h3>Packet decisions and influence log</h3><p>Each row records packet evidence, model confidence, host pressure, sensor quality, network load, selected model, and execution location.</p></div></div><div className="live-prediction-list">{result.predictions?.map((item) => <article className="live-prediction-row" key={item.packet_index}><strong>#{item.packet_index + 1}</strong><span className="prediction-tier">{tierLabels[item.selected_model] || item.selected_model}</span><b>{item.prediction?.replaceAll("_", " ") || "—"}</b><span>Risk {pct(item.risk?.score)}</span><small>{item.inference_location === "esp32_usb" ? `${item.device_inference_us ?? "—"} µs on ESP32` : `${Number(item.inference_ms || 0).toFixed(3)} ms on laptop`}</small><small>Confidence {pct(item.confidence)} · host stress {pct(item.host_resources?.resource_stress)} · edge load {pct(item.decision?.network_load_vs_edge_capacity)}</small><small>{sensorDecisionSummary(item.decision?.sensor_context)}</small><small>{item.decision?.execution_reason || item.decision?.reason}</small></article>)}</div></div>
    </>}

    <div className="panel live-hardware-panel">
      <div className="section-heading"><div><h2>ESP32 and sensor status</h2><p>Periodic USB sensor frames feed SARA before live packet decisions; qualified supply voltage/current can change execution placement.</p></div><span className={`connection-badge ${sensorsComplete ? "connected" : "checking"}`}><i />{sensor ? sensorsComplete ? "Sensors detected" : "Partial sensor report" : "Waiting for telemetry"}</span></div>
      {sensor?.voltage_v != null && (Number(sensor.voltage_v) < 2.7 || Number(sensor.voltage_v) > 5.5) && <div className="sensor-quality-note"><strong>INA219 reading is not qualified for routing</strong><span>{Number(sensor.voltage_v).toFixed(3)} V is outside the 2.7–5.5 V plausibility range. SARA records it but does not use it to switch execution.</span></div>}
      {sensor ? <div className="metric-grid live-metric-grid">{[
        ["Voltage", sensor.voltage_v == null ? "—" : `${Number(sensor.voltage_v).toFixed(3)} V`, "INA219"],
        ["Current", sensor.current_a == null ? "—" : `${(sensor.current_a * 1000).toFixed(2)} mA`, "INA219"],
        ["Temperature", sensor.temperature_ambient_c == null ? "—" : `${Number(sensor.temperature_ambient_c).toFixed(1)} °C`, "DHT22"],
        ["Humidity", sensor.humidity_pct == null ? "—" : `${Number(sensor.humidity_pct).toFixed(1)}%`, "DHT22"],
        ["Vibration", sensor.vibration ? "Detected" : "Normal", "SW-420"],
        ["Movement", sensor.movement ? "Detected" : "Stable", "MPU6500"],
        ["Acceleration", [sensor.accel_x_g, sensor.accel_y_g, sensor.accel_z_g].some((value) => value == null) ? "—" : `${Number(sensor.accel_x_g).toFixed(2)}, ${Number(sensor.accel_y_g).toFixed(2)}, ${Number(sensor.accel_z_g).toFixed(2)} g`, "MPU6500 · X/Y/Z"],
        ["Buzzer", sensor.alert ? "Active" : "Normal", "ESP32 output"],
      ].map(([label, value, source]) => <div className="metric-card live-metric" key={label}><span className="metric-copy"><span>{label}</span><strong>{value}</strong><small>{source}</small></span></div>)}</div> : <p className="empty-note">Connect the board and run live inference or the replay check to read its sensors. These sensor values are displayed separately; they are not IDS model features.</p>}
      <div className="live-replay-row"><span><Cpu size={15} /> Hardware check with 100 saved held-out rows</span><button className="secondary-button" disabled={backend.usbRunning || backend.status !== "connected" || !device} onClick={backend.runUsbReplay}>{backend.usbRunning ? "Running hardware check…" : "Run USB replay check"}</button></div>
      <div className="live-replay-row"><span><Activity size={15} /> Paired SARA ablation, same 100 labeled packets</span><button className="secondary-button" disabled={backend.usbRunning || backend.status !== "connected" || !device} onClick={backend.runSensorAblation}>{backend.usbRunning ? "Running paired ablation…" : "Run sensor ablation"}</button></div>
      {backend.ablationResult && <div className="ablation-result"><h3>Sensor input ablation</h3><p>{backend.ablationResult.samples} paired packet rows · {backend.ablationResult.comparison_summary?.with_physical_sensor_context?.qualified_sensor_samples || 0} qualified INA219 samples</p><div className="ablation-grid">{[["Without physical inputs", backend.ablationResult.comparison_summary?.without_physical_sensor_inputs], ["With physical context", backend.ablationResult.comparison_summary?.with_physical_sensor_context]].map(([label, arm]) => <article className="metric-card" key={label}><small>{label}</small><strong>{pct(arm?.accuracy)}</strong><span>Accuracy · macro-F1 {pct(arm?.macro_f1)}</span><span>ESP32 {arm?.execution_locations?.esp32_usb || 0} · laptop {arm?.execution_locations?.local_host || 0} · remote {arm?.execution_locations?.trusted_remote || 0}</span></article>)}</div><p>{backend.ablationResult.interpretation}</p><p>Full paired decision traces saved to {backend.ablationResult.artifact}</p></div>}
    </div>

    <div className="panel live-hardware-panel">
      <div className="section-heading"><div><h2>Networked federated learning</h2><p>Separate computer nodes train locally and exchange model updates. This does not alter the ESP32 sensor or USB inference path.</p></div><span className={`connection-badge ${backend.networkedFederated?.status === "running" ? "connected" : "checking"}`}><i />{backend.networkedFederated?.status || "API unavailable"}</span></div>
      {backend.networkedFederated?.status === "running" || backend.networkedFederated?.status === "complete" || backend.networkedFederated?.status === "timed_out" ? <div className="federated-rounds"><div><strong>Round {backend.networkedFederated.round}/{backend.networkedFederated.rounds}</strong><span>Updates received {backend.networkedFederated.updates_received ?? 0}</span><span>Nodes {backend.networkedFederated.nodes?.join(", ")}</span><span>Checkpoint {backend.networkedFederated.model_path}</span></div>{backend.networkedFederated.history?.map((round) => <div key={round.round}><strong>Round {round.round}</strong><span>Participants {round.participating_nodes?.join(", ")}</span><span>Mean local loss {Number(round.mean_local_loss).toFixed(4)}</span><span>Samples {Object.values(round.samples_per_node || {}).reduce((sum, count) => sum + count, 0)}</span></div>)}</div> : <p className="empty-note">No networked run is active. Configure coordinator and per-node tokens, then start it through the authenticated API endpoint documented in README.md. SARA requires fresh live risk/confidence and measured link metrics before it approves a run. The existing button above remains the local simulation.</p>}
      {backend.networkedFederated?.promotion_reason && <p className="empty-note">SARA model promotion: {backend.networkedFederated.promotion_approved ? "approved for the high laptop tier" : backend.networkedFederated.promotion_reason}</p>}
      <p className="empty-note">Only weights and sample counts are sent. This implementation has no secure aggregation or differential privacy. The global MLP replaces the laptop high tier only after passing the held-out validation gate; the ESP32 low tier remains unchanged and is not an FL training client.</p>
    </div>
  </>;
}
