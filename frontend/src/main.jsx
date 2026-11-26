import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity, Bell, Check, ChevronRight, Cpu, Gauge, LayoutDashboard, Menu,
  Moon, Network, Pause, Play, Radio, Server, ShieldAlert, ShieldCheck, Sparkles,
  Sun, Trash2, UserRound, X, Zap,
} from "lucide-react";
import GradientButton from "./components/kokonutui/gradient-button";
import { events, measured, models, readLocal, writeLocal } from "./data";
import useBackendApi from "./hooks/useBackendApi";
import LiveIdsPage from "./pages/LiveIdsPage";
import ModelTiersPage from "./pages/ModelTiersPage";
import "./styles.css";

const pages = [
  { id: "sara", label: "Live IDS", icon: ShieldCheck },
  { id: "models", label: "Model tiers", icon: Cpu },
];
const incidents = events.filter((event) => ["HIGH", "CRITICAL"].includes(event.severity));
const notificationEvents = [...incidents].reverse();
const latestModel = events.at(-1)?.selected_model?.toLowerCase() ?? models[0]?.key ?? "";

function App() {
  const backend = useBackendApi();
  const [page, setPage] = useState(() => {
    const requestedPage = window.location.hash.slice(1);
    return pages.some((item) => item.id === requestedPage) ? requestedPage : "sara";
  });
  const [profile, setProfile] = useState(() => {
    const saved = readLocal("adaptive-profile", null);
    return saved && typeof saved === "object" && !Array.isArray(saved)
      ? { name: typeof saved.name === "string" ? saved.name : "", email: typeof saved.email === "string" ? saved.email : "" }
      : null;
  });
  const [devices, setDevices] = useState(() => {
    const saved = readLocal("adaptive-devices", []);
    return Array.isArray(saved) ? saved.filter((device) => device && typeof device.id === "string" && typeof device.name === "string") : [];
  });
  const [preferredModel, setPreferredModel] = useState(() => readLocal("adaptive-model", latestModel));
  const [readNotifications, setReadNotifications] = useState(() => {
    const saved = readLocal("adaptive-read-notifications", []);
    return Array.isArray(saved) ? saved.filter((eventId) => Number.isFinite(eventId)) : [];
  });
  const [theme, setTheme] = useState(() => readLocal("adaptive-theme", "dark") === "light" ? "light" : "dark");
  const [mqttSettings, setMqttSettings] = useState(() => {
    const saved = readLocal("adaptive-mqtt-settings", null);
    return saved && typeof saved === "object"
      ? { brokerUrl: typeof saved.brokerUrl === "string" ? saved.brokerUrl : "ws://localhost:9001/mqtt", deviceId: typeof saved.deviceId === "string" ? saved.deviceId : "esp32-ids-01" }
      : { brokerUrl: "ws://localhost:9001/mqtt", deviceId: "esp32-ids-01" };
  });
  const [selectedEventCount, setSelectedEventCount] = useState(Math.min(25, events.length));
  const [playing, setPlaying] = useState(false);
  const [replayIndex, setReplayIndex] = useState(-1);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [modal, setModal] = useState(null);
  const [notice, setNotice] = useState("");
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => { writeLocal("adaptive-profile", profile); }, [profile]);
  useEffect(() => { writeLocal("adaptive-devices", devices); }, [devices]);
  useEffect(() => { writeLocal("adaptive-model", preferredModel); }, [preferredModel]);
  useEffect(() => { writeLocal("adaptive-read-notifications", readNotifications); }, [readNotifications]);
  useEffect(() => { writeLocal("adaptive-theme", theme); }, [theme]);
  useEffect(() => { writeLocal("adaptive-mqtt-settings", mqttSettings); }, [mqttSettings]);
  useEffect(() => {
    if (!playing || !events.length) return undefined;
    const timer = window.setInterval(() => {
      setReplayIndex((current) => {
        const next = current + 1;
        return Math.min(next, events.length - 1);
      });
    }, 900);
    return () => window.clearInterval(timer);
  }, [playing]);
  useEffect(() => {
    if (playing && replayIndex >= events.length - 1) setPlaying(false);
  }, [playing, replayIndex]);
  useEffect(() => {
    if (!notice) return undefined;
    const timer = window.setTimeout(() => setNotice(""), 3200);
    return () => window.clearTimeout(timer);
  }, [notice]);
  useEffect(() => {
    const syncPage = () => {
      const requestedPage = window.location.hash.slice(1);
      setPage(pages.some((item) => item.id === requestedPage) ? requestedPage : "sara");
      setSelectedEvent(null);
    };
    window.addEventListener("popstate", syncPage);
    return () => window.removeEventListener("popstate", syncPage);
  }, []);

  const unreadCount = notificationEvents.filter((event) => !readNotifications.includes(event.event_id)).length;
  const activeModel = models.find((model) => model.key === preferredModel) ?? models.find((model) => model.key === latestModel);
  const visibleEvents = playing && replayIndex >= 0
    ? events.slice(Math.max(0, replayIndex - 7), replayIndex + 1).reverse()
    : [...events].slice(-8).reverse();

  const navigate = (nextPage) => {
    if (window.location.hash !== `#${nextPage}`) window.history.pushState({}, "", `#${nextPage}`);
    setPage(nextPage);
    setMobileOpen(false);
    setNotificationsOpen(false);
    setAccountOpen(false);
    setSelectedEvent(null);
  };
  const startReplay = () => {
    if (!events.length) {
      setNotice("No recorded experiment events are available.");
      return;
    }
    setReplayIndex(0);
    setPlaying(true);
    navigate("telemetry");
  };
  const markAllRead = () => {
    setReadNotifications(notificationEvents.map((event) => event.event_id));
  };
  const markRead = (eventId) => {
    setReadNotifications((current) => current.includes(eventId) ? current : [...current, eventId]);
  };

  return (
    <div className={`app-shell ${theme === "light" ? "light-theme" : ""}`}>
      {mobileOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
      <aside className={`sidebar ${mobileOpen ? "mobile-open" : ""}`}>
        <button className="brand" onClick={() => navigate("sara")} aria-label="Go to live IDS">
          <span className="brand-mark"><Sparkles size={18} /></span>
          <span><strong>Adaptive<span>ML</span></strong><small>IIoT command center</small></span>
        </button>
        <p className="nav-label">Workspace</p>
        <nav aria-label="Main navigation">
          {pages.map(({ id, label, icon: Icon }) => <button key={id} className={page === id ? "active" : ""} onClick={() => navigate(id)}>
            <Icon size={17} /><span>{label}</span><ChevronRight className="nav-chevron" size={14} />
          </button>)}
        </nav>
        <div className="sidebar-footer"><span className={`status-dot ${backend.status === "connected" ? "connected" : ""}`} /><span><strong>{backend.status === "connected" ? "IDS API connected" : "IDS API disconnected"}</strong><small>{backend.usbDevice?.ports?.find((port) => port.accessible)?.port || "ESP32 USB not detected"}</small></span></div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <button className="mobile-menu icon-button" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={19} /></button>
          <div className="crumb">AdaptiveML <span>/</span> {pages.find((item) => item.id === page)?.label}</div>
          <div className="top-actions"><span className={`connection-badge ${backend.status === "connected" ? "connected" : "checking"}`}><i />{backend.status === "connected" ? "API ready" : "API offline"}</span><span className={`connection-badge ${backend.usbDevice?.ports?.some((port) => port.accessible) ? "connected" : "checking"}`}><i />{backend.usbDevice?.ports?.find((port) => port.accessible)?.port || "ESP32 offline"}</span></div>
        </header>

        <section className="page-content">
          {page === "sara" && <LiveIdsPage backend={backend} />}
          {page === "models" && <ModelTiersPage apiModels={backend.models} />}
        </section>
      </main>

      {notice && <div className="toast" role="status"><Check size={15} />{notice}<button aria-label="Dismiss message" onClick={() => setNotice("")}><X size={14} /></button></div>}
      {modal && <AppModal modal={modal} profile={profile} onClose={() => setModal(null)} onSaveProfile={(nextProfile) => { setProfile(nextProfile); setModal(null); setNotice("Profile saved in this browser."); }} onAddDevice={(device) => { setDevices((current) => [device, ...current]); setModal(null); setNotice(`${device.name} added to this browser's device list.`); }} onRemoveDevice={(device) => { setDevices((current) => current.filter((item) => item.id !== device.id)); setModal(null); setNotice(`${device.name} removed from this browser's device list.`); }} />}
    </div>
  );
}

function OverviewPage({ events: records, activeModel, deviceCount, selectedEventCount, onCountChange, onNavigate, onReplay, onSelectEvent, backend }) {
  const sample = records.slice(-selectedEventCount);
  const criticalCount = sample.filter((event) => ["HIGH", "CRITICAL"].includes(event.severity)).length;
  const meanThreat = mean(sample.map((event) => event.threat_score));
  const meanLatency = mean(sample.map((event) => event.inference_latency_ms));
  const metricCards = [
    { label: "Recorded events", value: formatNumber(sample.length), note: "From saved experiment", icon: Activity, page: "telemetry" },
    { label: "High severity", value: formatNumber(criticalCount), note: "In selected event window", icon: ShieldAlert, page: "telemetry" },
    { label: "Mean inference time", value: `${meanLatency.toFixed(2)} ms`, note: "Measured in this sample", icon: Gauge, page: "models" },
    { label: "Registered here", value: formatNumber(deviceCount), note: "Saved in this browser", icon: Network, page: "devices" },
  ];
  const alerts = [...sample].filter((event) => ["HIGH", "CRITICAL"].includes(event.severity)).slice(-5).reverse();
  return <>
    <BackendInferencePanel backend={backend} onOpenPolicy={() => onNavigate("sara")} />
    <div className="page-intro"><div><p className="eyebrow"><span className="pulse" /> EXPERIMENT OVERVIEW</p><h1>Adaptive edge monitor</h1><p>Explore measured intrusion events and model performance from the saved simulation.</p></div><GradientButton label="Open event replay" variant="emerald" icon={<Play size={15} />} className="primary-button" onClick={onReplay} /></div>
    <div className="data-source-banner"><ShieldCheck size={15} /><span>Recorded experiment data</span><b>{formatNumber(measured.eventCount)} events</b><span className="banner-separator">·</span><span>mean threat score {meanThreat.toFixed(1)} / 100</span></div>
    <div className="metric-grid">{metricCards.map(({ label, value, note, icon: Icon, page }) => <button className="metric-card" key={label} onClick={() => onNavigate(page)}><span className="metric-icon"><Icon size={18} /></span><span className="metric-copy"><span>{label}</span><strong>{value}</strong><small>{note}</small></span><ChevronRight className="metric-chevron" size={15} /></button>)}</div>
    <div className="section-heading"><div><h2>Threat score across events</h2><p>Values from the selected portion of the experiment</p></div><div className="tabs">{[10, 25, records.length].filter((count, index, all) => count > 0 && all.indexOf(count) === index).map((count) => <button key={count} className={selectedEventCount === count ? "selected" : ""} onClick={() => onCountChange(count)}>{count === records.length ? "All" : `Last ${count}`}</button>)}</div></div>
    <div className="dashboard-grid"><div className="panel chart-panel"><PanelHeading title="Event threat scores" subtitle={`${sample.length} recorded events`} /><ThreatChart events={sample} /></div><div className="panel activity-panel"><PanelHeading title="Selected model" subtitle="Latest recorded inference" /><div className="model-summary"><span className="tier-orb"><Cpu size={19} /></span><span><small>Most recent event used</small><strong>{records.at(-1)?.selected_model || "No model data"}</strong></span><button className="quiet-button" onClick={() => onNavigate("models")}>Compare</button></div><div className="status-list"><Status label="Prediction match rate" value="From historical experiment replay" valueRight={`${(measured.matchedPredictions * 100).toFixed(1)}%`} /><Status label="Model profiles" value="Loaded from saved artifacts" valueRight={formatNumber(backend.models?.model_count || 0)} /><Status label="Registered devices" value="Stored in this browser" valueRight={formatNumber(deviceCount)} /></div></div></div>
    <div className="bottom-grid"><div className="panel"><div className="panel-title"><PanelHeading title="High severity events" subtitle="Select an event to inspect its recorded details" /><button className="text-button" onClick={() => onNavigate("telemetry")}>All events <ChevronRight size={14} /></button></div><div className="alerts">{alerts.map((event) => <EventRow key={event.event_id} event={event} onClick={() => onSelectEvent(event)} />)}{!alerts.length && <p className="empty-note">No high severity events in this selection.</p>}</div></div><div className="panel quick-panel"><PanelHeading title="Quick actions" subtitle="Open a real project view" /><button className="action" onClick={onReplay}><Play size={17} /><span><strong>Replay experiment</strong><small>Step through saved events</small></span><ChevronRight size={15} /></button><button className="action" onClick={() => onNavigate("devices")}><Network size={17} /><span><strong>Manage device list</strong><small>{formatNumber(deviceCount)} saved in this browser</small></span><ChevronRight size={15} /></button></div></div>
  </>;
}

function BackendInferencePanel({ backend, onOpenPolicy }) {
  const result = backend.latestInference;
  const modelCount = backend.models?.model_count;
  return <section className="panel api-inference-panel">
    <div className="api-panel-heading"><span><p className="eyebrow"><span className={`pulse ${backend.status === "connected" ? "" : "muted-pulse"}`} /> BACKEND INFERENCE</p><h2>Run a saved test sample through SARA</h2><p>Uses the checked-in six-class models and saved test split. This is replay data, not live network capture.</p></span><span className={`connection-badge ${backend.status === "connected" ? "connected" : backend.status}`}><i />{backend.status === "connected" ? "API connected" : backend.status}</span></div>
    <div className="api-panel-actions"><button className="primary-button" disabled={backend.status !== "connected" || backend.running} onClick={backend.runInference}>{backend.running ? "Running inference…" : "Run saved sample inference"}</button><button className="quiet-button" onClick={onOpenPolicy}>View SARA policy <ChevronRight size={14} /></button><span className="api-model-count">{modelCount == null ? "Model registry unavailable" : `${modelCount} trained model artifacts loaded`}</span></div>
    {result && <div className="api-result-row"><span className="metric-icon"><ShieldCheck size={17} /></span><span><small>Latest replay prediction</small><strong>{result.prediction} · {result.model}</strong><small>True label: {result.true_label} · sample {result.sample_index} · {Number(result.inference_ms).toFixed(3)} ms on this host</small></span><span className="api-result-risk">Risk {Number(result.adaptive?.risk?.score ?? 0).toFixed(2)}</span></div>}
    {backend.error && <p className="form-error" role="status">{backend.status === "disconnected" ? `Backend unavailable: ${backend.error}. Start it with python -m src.api.app.` : backend.error}</p>}
  </section>;
}

function SaraPage({ backend }) {
  const [captureInterface, setCaptureInterface] = useState("");
  const [captureAuthorized, setCaptureAuthorized] = useState(false);
  const [usbLiveAuthorized, setUsbLiveAuthorized] = useState(false);
  const [probeTarget, setProbeTarget] = useState("");
  const decision = backend.latestInference?.adaptive?.decision;
  const risk = backend.latestInference?.adaptive?.risk;
  const resource = backend.resources;
  const cards = [
    ["API mode", backend.health?.mode || "—", backend.health?.mode_source || "Waiting for API"],
    ["Current action", decision?.action || backend.adaptive?.current_model || "—", decision?.reason || "Run an inference to generate a decision"],
    ["Cross-layer risk", risk ? `${(risk.score * 100).toFixed(1)}%` : "—", risk?.explanation || "No risk assessment yet"],
    ["Model disagreement", risk?.model_disagreement == null ? "—" : `${(risk.model_disagreement * 100).toFixed(1)}%`, "Vote disagreement among all three saved models"],
    ["Host CPU", resource?.cpu_percent == null ? "Unavailable" : `${resource.cpu_percent.toFixed(1)}%`, resource?.source || "Waiting for API"],
    ["Host memory", resource?.memory_percent == null ? "Unavailable" : `${resource.memory_percent.toFixed(1)}%`, resource?.available_memory_mb == null ? "No memory reading" : `${resource.available_memory_mb.toFixed(0)} MB available`],
    ["Trusted offloading", backend.offloading?.available ? `${backend.offloading.trusted_nodes.length} node(s)` : "Not configured", backend.offloading?.reason || "Configured peers are used only when local constraints trigger offloading"],
  ];
  const latestCapture = backend.networkStats?.items?.find((item) => item.source === "capture");
  return <>
    <PageHeading eyebrow="SECURITY-AWARE RESOURCE ADAPTATION" title="SARA decisions" description="The policy uses validation performance to enforce a recall floor. Resource stress affects feasibility, not attack evidence by itself." />
    <div className="tier-grid sara-cards">{cards.map(([label, value, detail]) => <article className="panel sara-card" key={label}><small>{label}</small><strong>{value}</strong><p>{detail}</p></article>)}</div>
    <div className="panel sara-detail-panel"><PanelHeading title="Latest decision trace" subtitle={backend.latestInference ? `Saved test sample ${backend.latestInference.sample_index}` : "No inference has been run in this browser session"} />
      {decision ? <><div className="inference-values"><div><small>Selected model</small><strong>{backend.latestInference.model}</strong></div><div><small>Required macro recall</small><strong>{Number(decision.required_macro_recall).toFixed(2)}</strong></div><div><small>Validation recall</small><strong>{Number(decision.validation_macro_recall).toFixed(3)}</strong></div><div><small>Coverage status</small><strong>{decision.security_coverage_degraded ? "DEGRADED" : "Within floor"}</strong></div><div><small>Prediction confidence</small><strong>{backend.latestInference.model_diagnostics?.prediction_confidence == null ? "—" : percent(backend.latestInference.model_diagnostics.prediction_confidence)}</strong></div><div><small>Model votes</small><strong>{Object.values(backend.latestInference.model_diagnostics?.model_predictions || {}).join(" · ") || "—"}</strong></div></div><p className="sara-reason">{decision.reason}</p><p className="empty-note">Network evidence: {Number(risk.network_evidence).toFixed(3)} · Resource stress: {Number(risk.resource_stress).toFixed(3)} · Entropy: {Number(risk.model_uncertainty).toFixed(3)} · Model disagreement: {Number(risk.model_disagreement).toFixed(3)}</p>{decision.rejected_actions?.length > 0 && <div className="sara-rejected"><strong>Rejected actions</strong>{decision.rejected_actions.map((item) => <p key={item.action}>{item.action}: {item.reason}</p>)}</div>}</> : <div className="empty-state"><Server size={25} /><p>Connect the Python API and run a saved sample to inspect the real decision trace.</p></div>}
    </div>
    <div className="panel sara-detail-panel"><PanelHeading title="Network measurements" subtitle="Host rates, optional authorized capture, and explicit RTT probe" />
      <div className="inference-values"><div><small>Host traffic rate</small><strong>{resource?.network_rates?.bytes_per_second == null ? "Collecting baseline" : `${formatDecimal(resource.network_rates.bytes_per_second)} B/s`}</strong></div><div><small>Host packet rate</small><strong>{resource?.network_rates?.packets_per_second == null ? "—" : `${formatDecimal(resource.network_rates.packets_per_second)} pkt/s`}</strong></div><div><small>Observed bandwidth</small><strong>{resource?.network_rates?.bandwidth_bps == null ? "—" : `${formatDecimal(resource.network_rates.bandwidth_bps / 1e6)} Mbps`}</strong></div><div><small>Active probe</small><strong>{backend.networkProbe ? `${backend.networkProbe.target_ip} · ${backend.networkProbe.packets_received}/${backend.networkProbe.packets_sent} replies` : "Not run"}</strong></div></div>
      <form className="mqtt-fields" onSubmit={(event) => { event.preventDefault(); backend.probeNetwork({ targetIp: probeTarget, packets: 5 }); }}><label className="form-label">Target IP for RTT/loss/jitter probe<input inputMode="decimal" value={probeTarget} onChange={(event) => setProbeTarget(event.target.value)} placeholder="Enter an IP you are authorized to probe" /></label><button className="secondary-button" disabled={backend.networkRunning || !probeTarget.trim()} type="submit">{backend.networkRunning ? "Measuring…" : "Run 5-packet ping probe"}</button></form>
      {backend.networkProbe && <p className="empty-note">Measured round-trip latency: {backend.networkProbe.rtt_ms.mean == null ? "no replies" : `${formatDecimal(backend.networkProbe.rtt_ms.mean)} ms`} · loss: {backend.networkProbe.packet_loss_pct == null ? "unavailable" : `${formatDecimal(backend.networkProbe.packet_loss_pct)}%`} · jitter: {backend.networkProbe.jitter_ms == null ? "unavailable" : `${formatDecimal(backend.networkProbe.jitter_ms)} ms`}. {backend.networkProbe.note}</p>}
      <form className="mqtt-fields" onSubmit={(event) => { event.preventDefault(); backend.captureNetwork({ interfaceName: captureInterface, authorized: captureAuthorized }); }}><label className="form-label">Interface (blank uses OS default)<input value={captureInterface} onChange={(event) => setCaptureInterface(event.target.value)} placeholder="e.g. wlan0" /></label><label className="checkbox-label"><input type="checkbox" checked={captureAuthorized} onChange={(event) => setCaptureAuthorized(event.target.checked)} /> I am authorized to monitor this interface</label><button className="secondary-button" disabled={backend.networkRunning || !captureAuthorized} type="submit">{backend.networkRunning ? "Capturing for up to 10 seconds…" : "Capture up to 500 local packets"}</button></form>
      {latestCapture && <p className="empty-note">Latest authorized capture: {latestCapture.packets} packets · avg size {latestCapture.average_packet_size_bytes == null ? "—" : `${formatDecimal(latestCapture.average_packet_size_bytes)} B`} · {formatDecimal(latestCapture.packets_per_second)} pkt/s · {formatDecimal(latestCapture.bandwidth_bps / 1e6)} Mbps observed · TCP SYN rate {formatDecimal(latestCapture.connection_frequency_per_second)}/s · protocols {latestCapture.protocol_types.join(", ") || "none"} · utilization {latestCapture.bandwidth_utilization_pct == null ? "unknown (link capacity unavailable)" : `${formatDecimal(latestCapture.bandwidth_utilization_pct)}%`}. RTT/loss/jitter require the separate probe. These aggregates are not passed to the IDS model.</p>}
      {backend.error && <p className="form-error" role="status">{backend.error}</p>}
    </div>
    <div className="panel sara-detail-panel"><PanelHeading title="Offline ESP32 inference over USB" subtitle="No Wi-Fi or MQTT required" />
      <p>The laptop captures up to 100 TCP/UDP packet headers for five seconds, applies the saved training transform, then sends feature vectors over USB. The ESP32 runs the fixed-point decision tree and returns predictions. Payloads and endpoint addresses are not retained; Wi-Fi is not used. Live predictions have no ground-truth labels, so accuracy is not reported.</p>
      <div className="inference-values"><div><small>Detected serial port</small><strong>{backend.usbDevice?.ports?.[0]?.port || "Not detected"}</strong></div><div><small>Port access</small><strong>{backend.usbDevice?.ports?.[0]?.accessible ? "Available" : backend.usbDevice?.ports?.length ? "Permission denied" : "—"}</strong></div><div><small>Transport</small><strong>USB · 115200 baud</strong></div><div><small>Board Wi-Fi</small><strong>Not used</strong></div></div>
      <button className="secondary-button" disabled={backend.usbRunning || backend.status !== "connected" || !backend.usbDevice?.ports?.length || !backend.usbDevice?.ports?.[0]?.accessible} onClick={backend.runUsbReplay}>{backend.usbRunning ? "Running 100 samples on ESP32…" : "Run 100 samples on ESP32"}</button>
      <form className="mqtt-fields" onSubmit={(event) => { event.preventDefault(); backend.runUsbLiveCapture({ interfaceName: captureInterface, authorized: usbLiveAuthorized }); }}><label className="checkbox-label"><input type="checkbox" checked={usbLiveAuthorized} onChange={(event) => setUsbLiveAuthorized(event.target.checked)} /> I am authorized to monitor this interface and explicitly approve this capture now</label><button className="secondary-button" disabled={backend.usbRunning || backend.status !== "connected" || !usbLiveAuthorized || !backend.usbDevice?.ports?.some((port) => port.accessible)} type="submit">{backend.usbRunning ? "Capturing and sending packets…" : "Capture packets and infer on ESP32"}</button></form>
      {backend.usbResult?.device_telemetry && (() => {
        const telemetry = backend.usbResult.device_telemetry;
        const sensorCards = [
          ["Voltage", telemetry.voltage_v == null ? "—" : `${formatDecimal(telemetry.voltage_v)} V`, "INA219"],
          ["Current", telemetry.current_a == null ? "—" : `${formatDecimal(telemetry.current_a * 1000)} mA`, "INA219"],
          ["Power", telemetry.power_w == null ? "—" : `${formatDecimal(telemetry.power_w)} W`, "INA219 · raw direction retained"],
          ["Energy", telemetry.energy_j == null ? "—" : `${formatDecimal(telemetry.energy_j)} J`, "INA219 integration"],
          ["Temperature", telemetry.temperature_ambient_c == null ? "—" : `${formatDecimal(telemetry.temperature_ambient_c)} °C`, "DHT22"],
          ["Humidity", telemetry.humidity_pct == null ? "—" : `${formatDecimal(telemetry.humidity_pct)}%`, "DHT22"],
          ["Vibration", telemetry.vibration ? "DETECTED" : "Normal", "SW-420"],
          ["Movement", telemetry.movement ? "DETECTED" : "Stable", "MPU6500"],
          ["Acceleration", [telemetry.accel_x_g, telemetry.accel_y_g, telemetry.accel_z_g].some((value) => value == null) ? "—" : `${formatDecimal(telemetry.accel_x_g)}, ${formatDecimal(telemetry.accel_y_g)}, ${formatDecimal(telemetry.accel_z_g)} g`, "MPU6500 · X / Y / Z"],
          ["Buzzer alert", telemetry.alert ? "ACTIVE" : "Normal", "Physical board output"],
        ];
        return <><div className="section-heading"><div><h3>Physical sensor readings</h3><p>Measured on the ESP32 and returned over USB · {new Date(telemetry.timestamp * 1000).toLocaleTimeString()}</p></div><span className={`connection-badge ${telemetry.quality === "good" ? "connected" : "checking"}`}><i />{telemetry.quality}</span></div><div className="metric-grid live-metric-grid">{sensorCards.map(([label, value, source]) => <div className="metric-card live-metric" key={label}><span className="metric-copy"><span>{label}</span><strong>{value}</strong><small>{source}</small></span></div>)}</div><p className="empty-note">OLED capability: {telemetry.sensor_status?.oled ? "detected" : "not detected"} · onboard alert uses vibration or temperature ≥ 40 °C. Sensors stay out of the IDS features; SARA uses qualified supply voltage for ESP32 execution placement and logs ambient/motion as non-security context.</p></>;
      })()}
      {backend.usbResult && <div className="federated-rounds"><div><strong>{backend.usbResult.mode?.includes("live") ? "ESP32 live packet predictions" : "ESP32 replay result"}</strong>{backend.usbResult.accuracy == null ? <span>Accuracy unavailable for unlabeled live packets</span> : <><span>Accuracy {percent(backend.usbResult.accuracy)}</span><span>Macro F1 {percent(backend.usbResult.macro_f1)}</span></>}<span>Packets {backend.usbResult.samples}</span></div><div><strong>Measured on this run</strong><span>Device p95 {formatDecimal(backend.usbResult.device_inference_us.p95)} µs</span><span>USB total {formatDecimal(backend.usbResult.usb_round_trip_ms.total)} ms</span><span>Energy from INA219 since boot: {formatDecimal(backend.usbResult.device_telemetry?.energy_j)} J</span></div>{backend.usbResult.predictions?.map((item) => <span key={item.packet_index}>Packet {item.packet_index + 1}: {item.prediction}</span>)}</div>}
      {backend.error && <p className="form-error" role="status">{backend.error}</p>}
      <p className="empty-note">Flash <code>hardware/esp32_usb_ids/esp32_usb_ids.ino</code> first. This path uses the compatible lightweight tree; the other sklearn tiers are laptop models.</p>
    </div>
    {backend.federated && <div className="panel sara-detail-panel"><PanelHeading title="Federated learning" subtitle={backend.federated.algorithm} /><p>{backend.federated.message || (backend.federated.simulated ? "Client partitions are simulated locally; no independent network clients are connected." : "Federated service unavailable.")}</p>{backend.federated.history?.length > 0 && <div className="federated-rounds">{backend.federated.history.map((round) => <div key={round.round}><strong>Round {round.round}</strong><span>Accuracy {percent(round.global_accuracy)}</span><span>Macro F1 {percent(round.global_macro_f1)}</span><span>Mean local loss {formatDecimal(round.mean_local_training_loss)}</span><span>Selected clients {round.selected_clients?.length ?? 0}/{Object.keys(round.client_availability || {}).length}</span><span>Model age {Object.keys(round.model_freshness?.client_model_age_rounds || {}).length ? "0 rounds (simulated)" : "—"}</span><span>Simulated comm. {formatSize(round.simulated_total_communication_bytes || 0)}</span><span>{round.accepted ? "Accepted" : "Rolled back"}</span></div>)}</div>}<button className="secondary-button" disabled={backend.federated.running || backend.status !== "connected"} onClick={backend.startFederatedTraining}>{backend.federated.running ? `Training round ${backend.federated.round}/${backend.federated.rounds}…` : "Run 3-round FedAvg simulation"}</button><p className="empty-note">Local loss and serialized communication bytes are measured in the simulation; client freshness, availability, and data partitions are simulated. No network transfer, real client availability, or energy is measured.</p></div>}
  </>;
}

function TelemetryPage({ events: records, visibleEvents, playing, replayIndex, selectedEvent, mqtt, settings, onSettingsChange, onToggle, onSelectEvent }) {
  return <>
    <PageHeading eyebrow="LIVE ESP32 TELEMETRY" title="Connect your sensor node." description="Receive real sensor readings from the ESP32 over MQTT. Connect to the broker’s WebSocket listener below." />
    <MqttConnectionPanel mqtt={mqtt} settings={settings} onSettingsChange={onSettingsChange} />
    {mqtt.telemetry || mqtt.inference ? <>
      <div className="data-source-banner"><Radio size={15} /><span>Live MQTT data</span><b>{mqtt.deviceId}</b><span className="banner-separator">·</span><span>{mqtt.lastMessageAt ? `Last message ${new Date(mqtt.lastMessageAt).toLocaleTimeString()}` : "Waiting for first message"}</span></div>
      <div className="metric-grid live-metric-grid">{[
        { label: "Temperature", value: mqtt.telemetry?.temperature_c == null ? "—" : `${formatDecimal(mqtt.telemetry.temperature_c)} °C`, unit: "DHT22", icon: Gauge },
        { label: "Humidity", value: mqtt.telemetry?.humidity_pct == null ? "—" : `${formatDecimal(mqtt.telemetry.humidity_pct)}%`, unit: "DHT22", icon: Activity },
        { label: "Vibration", value: mqtt.telemetry?.vibration == null ? "—" : mqtt.telemetry.vibration ? "DETECTED" : "Normal", unit: "SW-420", icon: ShieldAlert },
        { label: "Movement", value: mqtt.telemetry?.movement == null ? "—" : mqtt.telemetry.movement ? "DETECTED" : "Stable", unit: "MPU-6500", icon: Network },
        { label: "Acceleration", value: mqtt.telemetry?.accel_x_g == null ? "—" : `${formatDecimal(mqtt.telemetry.accel_x_g)}, ${formatDecimal(mqtt.telemetry.accel_y_g)}, ${formatDecimal(mqtt.telemetry.accel_z_g)} g`, unit: "MPU-6500 X / Y / Z", icon: Gauge },
        { label: "Alert", value: mqtt.telemetry?.alert == null ? "—" : mqtt.telemetry.alert ? "ACTIVE" : "Normal", unit: "Buzzer status", icon: Zap },
      ].map(({ label, value, unit, icon: Icon }) => <div className="metric-card live-metric" key={label}><span className="metric-icon"><Icon size={18} /></span><span className="metric-copy"><span>{label}</span><strong>{value}</strong><small>{unit}</small></span></div>)}</div>
      <div className="panel inference-panel"><PanelHeading title="Latest device inference" subtitle={mqtt.inference ? `Received ${mqtt.lastMessageAt ? new Date(mqtt.lastMessageAt).toLocaleTimeString() : "just now"}` : "No inference message received yet"} />{mqtt.inference ? <div className="inference-values"><div><small>Prediction</small><strong>{mqtt.inference.prediction || "—"}</strong></div><div><small>Severity</small><strong className={`severity-label ${(mqtt.inference.severity || "").toLowerCase()}`}>{mqtt.inference.severity || "—"}</strong></div><div><small>Threat score</small><strong>{mqtt.inference.threat_score == null ? "—" : `${formatDecimal(mqtt.inference.threat_score)} / 100`}</strong></div><div><small>Inference latency</small><strong>{mqtt.inference.latency_ms == null ? "—" : `${formatDecimal(mqtt.inference.latency_ms)} ms`}</strong></div></div> : <p className="empty-note">The device’s inference topic has not published a message.</p>}</div>
    </> : <div className="panel empty-state live-empty"><Radio size={26} /><h2>{mqtt.status === "connected" ? "Waiting for ESP32 messages" : "No live sensor data yet"}</h2><p>{mqtt.status === "connected" ? `Subscribed to esp32/${mqtt.deviceId}/telemetry and /inference. Power the device and check that it is publishing.` : "Connect to your MQTT broker above. Readings appear here only after the physical device publishes them."}</p></div>}
    <div className="section-heading"><div><h2>Recorded experiment replay</h2><p>Saved project data is separate from the live device stream.</p></div></div>
    <div className="replay-toolbar"><span className={`replay-state ${playing ? "is-playing" : ""}`}><i />{playing ? "Replay running" : "Replay paused"}</span><span>{replayIndex >= 0 ? `${replayIndex + 1} / ${records.length}` : `${records.length} saved events`}</span><button className="primary-button" onClick={onToggle}>{playing ? <Pause size={15} /> : <Play size={15} />}{playing ? "Pause replay" : replayIndex >= 0 ? "Resume replay" : "Start replay"}</button></div>
    <div className="telemetry-layout"><div className="panel event-list"><PanelHeading title="Recorded events" subtitle={`${records.length} events in the source experiment`} />{visibleEvents.map((event) => <EventRow key={event.event_id} event={event} active={selectedEvent?.event_id === event.event_id} onClick={() => onSelectEvent(event)} />)}</div><div className="panel event-detail"><PanelHeading title="Event details" subtitle={selectedEvent ? `Event ${selectedEvent.event_id}` : "Choose an event from the list"} />{selectedEvent ? <EventDetails event={selectedEvent} /> : <div className="empty-state"><Activity size={25} /><p>Select an event to inspect its prediction, severity, battery, model, and latency.</p></div>}</div></div>
  </>;
}

function MqttConnectionPanel({ mqtt, settings, onSettingsChange }) {
  const busy = ["connecting", "connected", "reconnecting"].includes(mqtt.status);
  const update = (field, value) => onSettingsChange((current) => ({ ...current, [field]: value }));
  return <form className="panel mqtt-connect-panel" onSubmit={(event) => { event.preventDefault(); mqtt.connect(settings); }}>
    <div className="mqtt-connect-heading"><span><h2>MQTT connection</h2><p>Use the broker’s WebSocket URL in this browser; the ESP32 uses MQTT port 1883.</p></span><span className={`connection-badge ${mqtt.status}`}><i />{mqtt.status}</span></div>
    <div className="mqtt-fields"><label className="form-label">Broker WebSocket URL<input value={settings.brokerUrl} disabled={busy} onChange={(event) => update("brokerUrl", event.target.value)} placeholder="ws://localhost:9001/mqtt" /></label><label className="form-label">ESP32 device ID<input value={settings.deviceId} disabled={busy} onChange={(event) => update("deviceId", event.target.value)} placeholder="esp32-ids-01" /></label><div className="mqtt-actions">{busy ? <button type="button" className="danger-button" onClick={mqtt.disconnect}>Disconnect</button> : <button type="submit" className="primary-button"><Radio size={15} />Connect device</button>}</div></div>
    {mqtt.error && <p className="form-error" role="alert">{mqtt.error}</p>}
    <p className="mqtt-help">Default local setup: <code>ws://localhost:9001/mqtt</code>. The broker must have WebSockets enabled. Do not use <code>localhost</code> as the broker address in the ESP32 firmware—use your computer’s LAN IP.</p>
  </form>;
}

function ModelsPage({ activeModel, apiModels, onSelect }) {
  return <>
    <PageHeading eyebrow="MEASURED MODEL RESULTS" title="Compare model tiers." description="Metrics below are loaded from the benchmark files committed with this project." />
    <div className="section-heading"><div><h2>Loaded artifacts · validation split</h2><p>Metrics computed from the saved 2,500-row validation split when the API starts. Latency is measured on the API host.</p></div></div>
    {apiModels?.models && <div className="tier-grid">{Object.entries(apiModels.models).map(([key, model]) => <article className="panel tier-card" key={key}><div className="tier-card-heading"><span className="metric-icon"><Cpu size={18} /></span><span className="healthy">{model.name}</span></div><div className="model-stat"><span>Accuracy</span><strong>{percent(model.metrics.accuracy)}</strong></div><div className="model-stat"><span>Macro precision</span><strong>{percent(model.metrics.macro_precision)}</strong></div><div className="model-stat"><span>Macro recall</span><strong>{percent(model.metrics.macro_recall)}</strong></div><div className="model-stat"><span>Macro F1</span><strong>{percent(model.metrics.macro_f1)}</strong></div><div className="model-stat"><span>Measured p95 host latency</span><strong>{formatDecimal(model.latency_ms.p95)} ms</strong></div><div className="model-stat"><span>Artifact size</span><strong>{formatSize(model.artifact_size_bytes / 1024)}</strong></div><small className="local-state-note">{model.evaluation.samples} validation records · energy is a historical estimate, not a measurement</small></article>)}</div>}
    {!apiModels && <div className="data-source-banner"><Server size={14} /><span>Python API unavailable; live artifact profiling is not loaded.</span></div>}
    <div className="section-heading"><div><h2>Saved historical benchmark</h2><p>Previous committed results; latency and energy mix measured and estimated values.</p></div></div>
    <div className="data-source-banner"><Cpu size={15} /><span>Preferred tier</span><b>{activeModel?.name || "Not selected"}</b><span className="banner-separator">·</span><span>Preference is saved in this browser</span></div>
    <div className="tier-grid">{models.map((model) => <article className={`panel tier-card ${activeModel?.key === model.key ? "active-tier" : ""}`} key={model.key}><div className="tier-card-heading"><span className="metric-icon"><Cpu size={18} /></span>{activeModel?.key === model.key && <span className="healthy"><Check size={12} /> Preferred</span>}</div><h2>{model.name}</h2><div className="model-stat"><span>Accuracy</span><strong>{percent(model.accuracy)}</strong></div><div className="model-stat"><span>Macro F1</span><strong>{percent(model.f1)}</strong></div><div className="model-stat"><span>Estimated latency</span><strong>{formatDecimal(model.latency)} ms</strong></div><div className="model-stat"><span>Model size</span><strong>{formatSize(model.size)}</strong></div><button className="secondary-button" disabled={activeModel?.key === model.key} onClick={() => onSelect(model)}>{activeModel?.key === model.key ? "Current preference" : "Set preferred tier"}</button></article>)}</div>
    <p className="local-state-note">Changing this preference updates this browser only. Connecting it to physical devices requires the project’s MQTT/backend connection.</p>
  </>;
}

function DevicesPage({ devices, onAdd, onRemove }) {
  return <>
    <div className="page-intro"><PageHeading eyebrow="BROWSER DEVICE LIST" title="Manage device records." description="These records are stored in this browser. They do not represent a live hardware connection." /><button className="primary-button" onClick={onAdd}><Network size={15} /> Add device</button></div>
    <div className="data-source-banner"><Network size={15} /><span>Saved device records</span><b>{formatNumber(devices.length)}</b><span className="banner-separator">·</span><span>Local to this browser</span></div>
    {devices.length ? <div className="panel device-table"><div className="device-table-heading"><span>Device</span><span>Connection details</span><span>Added</span><span /></div>{devices.map((device) => <div className="device-row" key={device.id}><span className="device-name"><span className="metric-icon"><Network size={16} /></span><strong>{device.name}</strong></span><span>{device.connection || "No connection details"}</span><time dateTime={device.createdAt}>{new Date(device.createdAt).toLocaleDateString()}</time><button className="remove-device" aria-label={`Remove ${device.name}`} onClick={() => onRemove(device)}><Trash2 size={15} /> Remove</button></div>)}</div> : <div className="panel empty-state device-empty"><Network size={28} /><h2>No device records yet</h2><p>Add the devices you want to keep track of in this browser.</p><button className="secondary-button" onClick={onAdd}>Add first device</button></div>}
  </>;
}

function ResearchPage({ backend }) {
  return <>
    <PageHeading eyebrow="CURRENT EVIDENCE" title="Research results." description="Current saved-artifact evaluation and implementation limits. Historical simulation claims are excluded from these figures." />
    <div className="data-source-banner"><Cpu size={15} /><span>Evaluation</span><b>Fixed held-out test split · no retraining</b><span className="banner-separator">·</span><span>5,001 samples · energy and device RAM not measured</span></div>
    {backend.models?.models ? <div className="tier-grid">{Object.entries(backend.models.models).map(([key, model]) => <article className="panel research-card" key={key}><span className="metric-icon"><Cpu size={18} /></span><h2>{model.name}</h2>{[["Validation accuracy", model.metrics.accuracy], ["Validation macro recall", model.metrics.macro_recall], ["Validation macro F1", model.metrics.macro_f1]].map(([label, value]) => <div className="model-stat" key={label}><span>{label}</span><strong>{percent(value)}</strong></div>)}<div className="model-stat"><span>Artifact size</span><strong>{formatSize(model.artifact_size_bytes / 1024)}</strong></div><div className="model-stat"><span>API host p95 latency</span><strong>{formatDecimal(model.latency_ms.p95)} ms</strong></div><small className="local-state-note">Validation profile; host timing is not ESP32 timing. Energy not measured.</small></article>)}</div> : <div className="data-source-banner"><Server size={14} /><span>Start the Python API to load artifact evaluation profiles.</span></div>}
    <div className="panel sara-detail-panel"><PanelHeading title="Evidence gaps" subtitle="What remains before deployment or a research claim" /><p>The policy has not been evaluated against controlled adaptive baselines. ESP32 model inference works for saved feature replay, but a live packet-to-model feature extractor is still missing. Federated learning currently runs as an in-process simulation without network clients or formal privacy guarantees. See the project audit and literature review in the repository.</p></div>
  </>;
}

function EventDetails({ event }) {
  const fields = [
    ["Prediction", event.predicted_class?.replaceAll("_", " ")],
    ["Ground truth", event.true_class?.replaceAll("_", " ")],
    ["Confidence", percent(event.confidence)],
    ["Threat score", `${event.threat_score} / 100`],
    ["Battery", percent(event.battery_pct / 100)],
    ["Selected model", event.selected_model],
    ["Inference latency", `${formatDecimal(event.inference_latency_ms)} ms`],
    ["Energy consumed", `${formatDecimal(event.energy_consumed_uj)} µJ`],
  ];
  return <><div className="event-severity"><span className={`severity-dot ${event.severity.toLowerCase()}`} />{event.severity}<span className="event-time">{event.timestamp}</span></div><dl className="detail-list">{fields.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value ?? "—"}</dd></div>)}</dl><p className="event-reason">{event.reason}</p></>;
}

function AppModal({ modal, profile, onClose, onSaveProfile, onAddDevice, onRemoveDevice }) {
  useEffect(() => {
    const closeOnEscape = (event) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);
  const remove = modal.type === "remove-device";
  const profileForm = modal.type === "profile";
  return <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}><section className="modal-card" role="dialog" aria-modal="true" aria-labelledby="modal-title"><button className="modal-close icon-button" aria-label="Close dialog" onClick={onClose}><X size={17} /></button>
    {remove ? <><p className="eyebrow">DEVICE RECORD</p><h2 id="modal-title">Remove {modal.device.name}?</h2><p>This deletes the saved record from this browser.</p><div className="modal-actions"><button className="secondary-button" onClick={onClose}>Keep record</button><button className="danger-button" onClick={() => onRemoveDevice(modal.device)}>Remove record</button></div></>
      : profileForm ? <ProfileForm profile={profile} onCancel={onClose} onSave={onSaveProfile} />
      : <DeviceForm onCancel={onClose} onSave={onAddDevice} />}
  </section></div>;
}

function ProfileForm({ profile, onCancel, onSave }) {
  const [name, setName] = useState(profile?.name ?? "");
  const [email, setEmail] = useState(profile?.email ?? "");
  const [error, setError] = useState("");
  return <form onSubmit={(event) => { event.preventDefault(); if (!name.trim()) { setError("Enter a display name."); return; } onSave({ name: name.trim(), email: email.trim() }); }}>
    <p className="eyebrow">BROWSER PROFILE</p><h2 id="modal-title">Your profile</h2><p>This profile is stored locally in this browser.</p>
    <label className="form-label">Display name<input autoFocus value={name} onChange={(event) => setName(event.target.value)} maxLength={80} /></label>
    <label className="form-label">Email<input type="email" value={email} onChange={(event) => setEmail(event.target.value)} maxLength={160} /></label>
    {error && <p className="form-error" role="alert">{error}</p>}<div className="modal-actions"><button type="button" className="secondary-button" onClick={onCancel}>Cancel</button><button type="submit" className="primary-button">Save profile</button></div>
  </form>;
}

function DeviceForm({ onCancel, onSave }) {
  const [name, setName] = useState("");
  const [connection, setConnection] = useState("");
  const [error, setError] = useState("");
  return <form onSubmit={(event) => {
    event.preventDefault();
    const cleanName = name.trim();
    if (!cleanName) { setError("Enter a device name or ID."); return; }
    onSave({ id: crypto.randomUUID(), name: cleanName, connection: connection.trim(), createdAt: new Date().toISOString() });
  }}>
    <p className="eyebrow">DEVICE RECORD</p><h2 id="modal-title">Add a device</h2><p>The record will be saved to this browser's device list.</p>
    <label className="form-label">Device name or ID<input autoFocus value={name} onChange={(event) => setName(event.target.value)} maxLength={100} placeholder="Enter a device name" /></label>
    <label className="form-label">Connection detail <span>Optional</span><input value={connection} onChange={(event) => setConnection(event.target.value)} maxLength={200} placeholder="MQTT topic, address, or note" /></label>
    {error && <p className="form-error" role="alert">{error}</p>}<div className="modal-actions"><button type="button" className="secondary-button" onClick={onCancel}>Cancel</button><button type="submit" className="primary-button">Save device</button></div>
  </form>;
}

function PageHeading({ eyebrow, title, description }) {
  return <div className="page-heading"><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{description}</p></div>;
}

function PanelHeading({ title, subtitle, children }) {
  return <div className="panel-title"><span><h2>{title}</h2><p>{subtitle}</p></span>{children}</div>;
}

function Status({ label, value, valueRight }) {
  return <div className="status-row"><span><strong>{label}</strong><small>{value}</small></span><b>{valueRight}</b></div>;
}

function EventRow({ event, onClick, active = false }) {
  return <button className={`event-row ${active ? "event-active" : ""}`} onClick={onClick}><span className={`severity-dot ${event.severity.toLowerCase()}`} /><span className="event-row-copy"><strong>{event.predicted_class.replaceAll("_", " ")}</strong><small>Event {event.event_id} · {event.timestamp} · threat {event.threat_score}</small></span><span className={`severity-label ${event.severity.toLowerCase()}`}>{event.severity}</span><ChevronRight size={14} /></button>;
}

function ThreatChart({ events: records }) {
  if (!records.length) return <div className="empty-note">No experiment events available.</div>;
  const points = records.map((event, index) => {
    const x = records.length === 1 ? 350 : index / (records.length - 1) * 700;
    const y = 210 - (Math.max(0, Math.min(100, event.threat_score)) / 100) * 190;
    return [x, y];
  });
  const line = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const fill = `${line} L700 220 L0 220 Z`;
  return <div className="chart"><div className="y-labels"><span>100</span><span>75</span><span>50</span><span>25</span><span>0</span></div><div className="chart-area"><div className="grid-lines" /><svg viewBox="0 0 700 220" preserveAspectRatio="none" role="img" aria-label="Threat score across selected recorded events"><defs><linearGradient id="threat-area" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#22d3ee" stopOpacity=".30" /><stop offset="1" stopColor="#22d3ee" stopOpacity="0" /></linearGradient></defs><path d={fill} fill="url(#threat-area)" /><path d={line} fill="none" stroke="#22d3ee" strokeWidth="3" vectorEffect="non-scaling-stroke" /></svg><div className="x-labels"><span>Event {records[0].event_id}</span><span>Event {records.at(-1).event_id}</span></div></div></div>;
}

function initials(name) {
  return name.trim().split(/\s+/).slice(0, 2).map((part) => part[0]?.toUpperCase()).join("");
}
function mean(values) { return values.length ? values.reduce((sum, value) => sum + Number(value || 0), 0) / values.length : 0; }
function formatNumber(value) { return new Intl.NumberFormat().format(value); }
function percent(value) { return `${(Number(value || 0) * 100).toFixed(1)}%`; }
function formatDecimal(value) { return Number(value || 0).toFixed(2); }
function formatSize(value) { return Number(value) < 1 ? `${Number(value).toFixed(2)} KB` : `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 })} KB`; }

createRoot(document.getElementById("root")).render(<App />);
