import simulationEvents from "../../results/experiments/simulation_results.json";
import benchmarkResults from "../../results/tables/model_benchmark.json";
import tinymlResults from "../../results/tables/tinyml_optimization.json";
import federatedResults from "../../results/tables/federated_comparison.json";

export const events = [...simulationEvents].sort((left, right) => left.event_id - right.event_id);
export const benchmarks = benchmarkResults;
export const tinyml = tinymlResults;
export const federated = federatedResults;

export const models = [
  ...Object.entries(benchmarks).map(([key, result]) => ({
    key,
    name: result.model_name,
    accuracy: result.accuracy,
    f1: result.f1_macro,
    latency: result.estimated_latency_ms,
    size: result.model_size_kb,
    energy: result.est_energy_uJ,
  })),
  {
    key: "tinyml",
    name: tinyml.model_name,
    accuracy: tinyml.quantized_accuracy,
    f1: tinyml.quantized_f1_macro,
    latency: tinyml.quantized_latency_ms,
    size: tinyml.quantized_size_kb,
    energy: benchmarks.lightweight.est_energy_uJ,
  },
];

export const measured = {
  eventCount: events.length,
  criticalCount: events.filter((event) => ["HIGH", "CRITICAL"].includes(event.severity)).length,
  averageThreat: average(events.map((event) => event.threat_score)),
  averageLatency: average(events.map((event) => event.inference_latency_ms)),
  matchedPredictions: events.length
    ? events.filter((event) => event.true_class === event.predicted_class).length / events.length
    : 0,
};

export function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

export function readLocal(key, fallback) {
  try {
    const saved = window.localStorage.getItem(key);
    return saved === null ? fallback : JSON.parse(saved);
  } catch {
    return fallback;
  }
}

export function writeLocal(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    return false;
  }
  return true;
}
