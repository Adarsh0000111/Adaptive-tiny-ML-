import { Cpu } from "lucide-react";

const tiers = [
  ["lightweight", "Low", "ESP32-S3 over USB"],
  ["medium", "Medium", "Laptop"],
  ["heavy", "High", "Laptop"],
];
const pct = (value) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;

export default function ModelTiersPage({ apiModels }) {
  return <>
    <div className="page-heading"><p className="eyebrow">TRAINED ON OFFICIAL EDGE-IIOTSET PACKETS</p><h1>Model tiers</h1><p>Live SARA routing selects a tier for each packet. The high tier is available only when the live model signals support escalation; it is not assumed to be more accurate.</p></div>
    {!apiModels ? <div className="panel empty-state"><p>Connect the local API to load the trained model artifacts.</p></div> : <>
      <div className="tier-grid">{tiers.map(([key, label, location]) => {
        const model = apiModels.models?.[key];
        if (!model) return null;
        return <article className="panel tier-card" key={key}>
          <div className="tier-card-heading"><span className="metric-icon"><Cpu size={18} /></span><span className="healthy">{location}</span></div>
          <h2>{label} tier</h2>
          <div className="model-stat"><span>Test accuracy</span><strong>{pct(model.test_metrics?.accuracy)}</strong></div>
          <div className="model-stat"><span>Test macro recall</span><strong>{pct(model.test_metrics?.macro_recall)}</strong></div>
          <div className="model-stat"><span>Test macro F1</span><strong>{pct(model.test_metrics?.macro_f1)}</strong></div>
          <div className="model-stat"><span>Model size</span><strong>{(model.artifact_size_bytes / 1024).toFixed(1)} KB</strong></div>
          <p className="local-state-note">{model.name} · trained model evaluation on held-out packet rows</p>
        </article>;
      })}</div>
      <div className="panel model-data-note"><h3>Input and evaluation</h3><p>18 packet-level features: TCP flags/length/port, UDP timing, DNS/HTTP/MQTT indicators, and protocol indicators. These are not reconstructed flows. The held-out split is random at packet level and the source has no session IDs, so it does not prove generalization to independent sessions or arbitrary traffic.</p><p>Live traffic has no labels; dashboard predictions are not accuracy measurements. Current model test metrics are shown above.</p></div>
    </>}
  </>;
}
