import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity, Bell, Check, ChevronRight, Cpu, Gauge, LayoutDashboard, Menu,
  Moon, Network, Pause, Play, Radio, ShieldAlert, ShieldCheck, Sparkles,
  Sun, Trash2, UserRound, X, Zap,
} from "lucide-react";
import GradientButton from "./components/kokonutui/gradient-button";
import { events, federated, measured, models, readLocal, tinyml, writeLocal } from "./data";
import useMqttTelemetry from "./hooks/useMqttTelemetry";
import "./styles.css";

const pages = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "telemetry", label: "Live telemetry", icon: Radio },
  { id: "models", label: "Model tiers", icon: Cpu },
  { id: "devices", label: "Device list", icon: Network },
  { id: "research", label: "Research results", icon: Gauge },
];
const incidents = events.filter((event) => ["HIGH", "CRITICAL"].includes(event.severity));
const notificationEvents = [...incidents].reverse();
const latestModel = events.at(-1)?.selected_model?.toLowerCase() ?? models[0]?.key ?? "";

function App() {
  const mqttConnection = useMqttTelemetry();
  const [page, setPage] = useState(() => {
    const requestedPage = window.location.hash.slice(1);
    return pages.some((item) => item.id === requestedPage) ? requestedPage : "overview";
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
      setPage(pages.some((item) => item.id === requestedPage) ? requestedPage : "overview");
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
        <button className="brand" onClick={() => navigate("overview")} aria-label="Go to overview">
          <span className="brand-mark"><Sparkles size={18} /></span>
          <span><strong>Adaptive<span>ML</span></strong><small>IIoT command center</small></span>
        </button>
        <p className="nav-label">Workspace</p>
        <nav aria-label="Main navigation">
          {pages.map(({ id, label, icon: Icon }) => <button key={id} className={page === id ? "active" : ""} onClick={() => navigate(id)}>
            <Icon size={17} /><span>{label}</span><ChevronRight className="nav-chevron" size={14} />
          </button>)}
        </nav>
        <div className="sidebar-footer"><span className={`status-dot ${mqttConnection.status === "connected" ? "connected" : ""}`} /><span><strong>{mqttConnection.status === "connected" ? "ESP32 connected" : "MQTT disconnected"}</strong><small>{mqttConnection.deviceId || "Configure in Live telemetry"}</small></span></div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <button className="mobile-menu icon-button" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={19} /></button>
          <div className="crumb">Security operations <span>/</span> {pages.find((item) => item.id === page)?.label}</div>
          <div className="top-actions">
            <div className="popover-anchor">
              <button className={`icon-button notification-trigger ${notificationsOpen ? "active" : ""}`} aria-label={`Notifications, ${unreadCount} unread`} aria-expanded={notificationsOpen} onClick={() => { setNotificationsOpen((open) => !open); setAccountOpen(false); }}>
                <Bell size={18} />{unreadCount > 0 && <i className="notification-count">{unreadCount > 9 ? "9+" : unreadCount}</i>}
              </button>
              {notificationsOpen && <div className="popover notification-popover">
                <div className="popover-heading"><span><strong>Experiment alerts</strong><small>{unreadCount} unread from recorded data</small></span><button className="quiet-button" onClick={markAllRead} disabled={!unreadCount}>Mark all read</button></div>
                {notificationEvents.slice(0, 8).map((event) => <button className={`notification-item ${readNotifications.includes(event.event_id) ? "read" : ""}`} key={event.event_id} onClick={() => { markRead(event.event_id); navigate("telemetry"); setSelectedEvent(event); }}>
                  <span className={`severity-dot ${event.severity.toLowerCase()}`} /><span><strong>{event.severity} · {event.predicted_class.replaceAll("_", " ")}</strong><small>Event {event.event_id} · threat {event.threat_score}</small></span><ChevronRight size={15} />
                </button>)}
                {!notificationEvents.length && <p className="empty-note">No high severity events in this experiment.</p>}
              </div>}
            </div>
            <div className="popover-anchor">
              <button className="account-trigger" aria-label="Open account profile" aria-expanded={accountOpen} onClick={() => { setAccountOpen((open) => !open); setNotificationsOpen(false); }}>
                <span className="avatar">{profile?.name?.trim() ? initials(profile.name) : <UserRound size={16} />}</span>
                <span className="account-label">{profile?.name?.trim() || "Set up profile"}</span>
              </button>
              {accountOpen && <div className="popover account-popover"><span className="account-popover-label">BROWSER PROFILE</span>
                <strong>{profile?.name?.trim() || "Profile not set up"}</strong><small>{profile?.email || "No email saved"}</small>
                <button className="popover-action" onClick={() => { setModal({ type: "profile" }); setAccountOpen(false); }}>Edit profile <ChevronRight size={15} /></button>
                <button className="popover-action" onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}>{theme === "dark" ? <Sun size={15} /> : <Moon size={15} />} Switch to {theme === "dark" ? "light" : "dark"} theme</button>
              </div>}
            </div>
          </div>
        </header>

        <section className="page-content">
          {page === "overview" && <OverviewPage events={events} activeModel={activeModel} deviceCount={devices.length} selectedEventCount={selectedEventCount} onCountChange={setSelectedEventCount} onNavigate={navigate} onReplay={startReplay} onSelectEvent={(event) => { navigate("telemetry"); setSelectedEvent(event); }} />}
          {page === "telemetry" && <TelemetryPage events={events} visibleEvents={visibleEvents} playing={playing} replayIndex={replayIndex} selectedEvent={selectedEvent} mqtt={mqttConnection} settings={mqttSettings} onSettingsChange={setMqttSettings} onToggle={() => {
            if (!playing && replayIndex >= events.length - 1) setReplayIndex(-1);
            if (!playing && replayIndex < 0) setReplayIndex(0);
            setPlaying((current) => !current);
          }} onSelectEvent={setSelectedEvent} />}
          {page === "models" && <ModelsPage activeModel={activeModel} onSelect={(model) => { setPreferredModel(model.key); setNotice(`${model.name} saved as your preferred model.`); }} />}
          {page === "devices" && <DevicesPage devices={devices} onAdd={() => setModal({ type: "device" })} onRemove={(device) => setModal({ type: "remove-device", device })} />}
          {page === "research" && <ResearchPage />}
        </section>
      </main>

      {notice && <div className="toast" role="status"><Check size={15} />{notice}<button aria-label="Dismiss message" onClick={() => setNotice("")}><X size={14} /></button></div>}
      {modal && <AppModal modal={modal} profile={profile} onClose={() => setModal(null)} onSaveProfile={(nextProfile) => { setProfile(nextProfile); setModal(null); setNotice("Profile saved in this browser."); }} onAddDevice={(device) => { setDevices((current) => [device, ...current]); setModal(null); setNotice(`${device.name} added to this browser's device list.`); }} onRemoveDevice={(device) => { setDevices((current) => current.filter((item) => item.id !== device.id)); setModal(null); setNotice(`${device.name} removed from this browser's device list.`); }} />}
    </div>
  );
}

function OverviewPage({ events: records, activeModel, deviceCount, selectedEventCount, onCountChange, onNavigate, onReplay, onSelectEvent }) {
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
    <div className="page-intro"><div><p className="eyebrow"><span className="pulse" /> EXPERIMENT OVERVIEW</p><h1>Adaptive edge monitor</h1><p>Explore measured intrusion events and model performance from the saved simulation.</p></div><GradientButton label="Open event replay" variant="emerald" icon={<Play size={15} />} className="primary-button" onClick={onReplay} /></div>
    <div className="data-source-banner"><ShieldCheck size={15} /><span>Recorded experiment data</span><b>{formatNumber(measured.eventCount)} events</b><span className="banner-separator">·</span><span>mean threat score {meanThreat.toFixed(1)} / 100</span></div>
    <div className="metric-grid">{metricCards.map(({ label, value, note, icon: Icon, page }) => <button className="metric-card" key={label} onClick={() => onNavigate(page)}><span className="metric-icon"><Icon size={18} /></span><span className="metric-copy"><span>{label}</span><strong>{value}</strong><small>{note}</small></span><ChevronRight className="metric-chevron" size={15} /></button>)}</div>
    <div className="section-heading"><div><h2>Threat score across events</h2><p>Values from the selected portion of the experiment</p></div><div className="tabs">{[10, 25, records.length].filter((count, index, all) => count > 0 && all.indexOf(count) === index).map((count) => <button key={count} className={selectedEventCount === count ? "selected" : ""} onClick={() => onCountChange(count)}>{count === records.length ? "All" : `Last ${count}`}</button>)}</div></div>
    <div className="dashboard-grid"><div className="panel chart-panel"><PanelHeading title="Event threat scores" subtitle={`${sample.length} recorded events`} /><ThreatChart events={sample} /></div><div className="panel activity-panel"><PanelHeading title="Selected model" subtitle="Latest recorded inference" /><div className="model-summary"><span className="tier-orb"><Cpu size={19} /></span><span><small>Most recent event used</small><strong>{records.at(-1)?.selected_model || "No model data"}</strong></span><button className="quiet-button" onClick={() => onNavigate("models")}>Compare</button></div><div className="status-list"><Status label="Prediction match rate" value="From experiment labels" valueRight={`${(measured.matchedPredictions * 100).toFixed(1)}%`} /><Status label="TinyML size reduction" value="Quantized vs original" valueRight={`${Number(tinyml.size_reduction_pct).toFixed(1)}%`} /><Status label="Registered devices" value="Stored in this browser" valueRight={formatNumber(deviceCount)} /></div></div></div>
    <div className="bottom-grid"><div className="panel"><div className="panel-title"><PanelHeading title="High severity events" subtitle="Select an event to inspect its recorded details" /><button className="text-button" onClick={() => onNavigate("telemetry")}>All events <ChevronRight size={14} /></button></div><div className="alerts">{alerts.map((event) => <EventRow key={event.event_id} event={event} onClick={() => onSelectEvent(event)} />)}{!alerts.length && <p className="empty-note">No high severity events in this selection.</p>}</div></div><div className="panel quick-panel"><PanelHeading title="Quick actions" subtitle="Open a real project view" /><button className="action" onClick={onReplay}><Play size={17} /><span><strong>Replay experiment</strong><small>Step through saved events</small></span><ChevronRight size={15} /></button><button className="action" onClick={() => onNavigate("devices")}><Network size={17} /><span><strong>Manage device list</strong><small>{formatNumber(deviceCount)} saved in this browser</small></span><ChevronRight size={15} /></button></div></div>
  </>;
}

function TelemetryPage({ events: records, visibleEvents, playing, replayIndex, selectedEvent, mqtt, settings, onSettingsChange, onToggle, onSelectEvent }) {
  return <>
    <PageHeading eyebrow="LIVE ESP32 TELEMETRY" title="Connect your sensor node." description="Receive real sensor readings from the ESP32 over MQTT. Connect to the broker’s WebSocket listener below." />
    <MqttConnectionPanel mqtt={mqtt} settings={settings} onSettingsChange={onSettingsChange} />
    {mqtt.telemetry || mqtt.inference ? <>
      <div className="data-source-banner"><Radio size={15} /><span>Live MQTT data</span><b>{mqtt.deviceId}</b><span className="banner-separator">·</span><span>{mqtt.lastMessageAt ? `Last message ${new Date(mqtt.lastMessageAt).toLocaleTimeString()}` : "Waiting for first message"}</span></div>
      <div className="metric-grid live-metric-grid">{[
        { label: "Battery", value: mqtt.telemetry?.battery_pct == null ? "—" : `${formatDecimal(mqtt.telemetry.battery_pct)}%`, unit: "from ESP32", icon: Zap },
        { label: "Bus voltage", value: mqtt.telemetry?.bus_voltage_v == null ? "—" : `${formatDecimal(mqtt.telemetry.bus_voltage_v)} V`, unit: "INA219", icon: Gauge },
        { label: "Shunt current", value: mqtt.telemetry?.shunt_current_ma == null ? "—" : `${formatDecimal(mqtt.telemetry.shunt_current_ma)} mA`, unit: "INA219", icon: Activity },
        { label: "Model tier", value: mqtt.telemetry?.model_tier || mqtt.inference?.model || "—", unit: "reported by device", icon: Cpu },
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

function ModelsPage({ activeModel, onSelect }) {
  return <>
    <PageHeading eyebrow="MEASURED MODEL RESULTS" title="Compare model tiers." description="Metrics below are loaded from the benchmark files committed with this project." />
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

function ResearchPage() {
  const energy = federated.energy_aware_fl;
  const standard = federated.standard_fl;
  return <>
    <PageHeading eyebrow="PROJECT RESULTS" title="Research at a glance." description="Summary values are read from the TinyML and federated-learning result files." />
    <div className="research-grid"><article className="panel research-card"><span className="metric-icon"><Zap size={18} /></span><h2>{tinyml.model_name}</h2><div className="model-stat"><span>Original size</span><strong>{formatSize(tinyml.original_size_kb)}</strong></div><div className="model-stat"><span>Quantized size</span><strong>{formatSize(tinyml.quantized_size_kb)}</strong></div><div className="model-stat"><span>Size reduction</span><strong>{percent(tinyml.size_reduction_pct / 100)}</strong></div><div className="model-stat"><span>Quantized accuracy</span><strong>{percent(tinyml.quantized_accuracy)}</strong></div><div className="model-stat"><span>ESP32 compatible</span><strong>{tinyml.esp32_compatible ? "Yes" : "No"}</strong></div></article><article className="panel research-card"><span className="metric-icon"><Network size={18} /></span><h2>Federated learning</h2>{[["Energy-aware accuracy", energy.final_global_accuracy], ["Standard accuracy", standard.final_global_accuracy], ["Energy-aware macro F1", energy.final_global_f1_macro], ["Average battery", energy.final_avg_battery_pct / 100]].map(([label, value]) => <div className="model-stat" key={label}><span>{label}</span><strong>{percent(value)}</strong></div>)}<div className="model-stat"><span>Communication overhead</span><strong>{formatDecimal(energy.total_comm_overhead_kb)} KB</strong></div></article></div>
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
