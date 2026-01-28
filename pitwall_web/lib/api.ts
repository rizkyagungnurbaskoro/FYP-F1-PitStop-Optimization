export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

async function fetchJson(path: string) {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return res.json();
}

export async function getSummary(mode: "strict" | "standard") {
  return fetchJson(`/metrics/summary?mode=${mode}`);
}

export async function getHoldout() {
  return fetchJson(`/metrics/holdout`);
}


export async function getShowcase(dataset: "my" | "ref") {
  return fetchJson(`/scenarios/showcase?dataset=${dataset}`);
}

export async function getScenario(dataset: "my" | "ref", id: string) {
  return fetchJson(`/scenario/${id}?dataset=${dataset}`);
}

/** New Granular Demo API */

export async function getDemoContext(dataset: "my" | "ref", driver?: string, circuit?: string, weather?: string, year?: string) {
  return fetch(`${API_BASE}/demo/context`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, driver, circuit, weather, year })
  }).then(res => res.json());
}

export async function getDemoDecision(selection: any) {
  return fetch(`${API_BASE}/demo/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(selection)
  }).then(res => res.json());
}

export async function getDemoTelemetry(selection: any) {
  return fetch(`${API_BASE}/demo/telemetry`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(selection)
  }).then(res => res.json());
}

export async function getDemoPitWindow(selection: any) {
  return fetch(`${API_BASE}/demo/pitwindow`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(selection)
  }).then(res => res.json());
}

export async function getDemoImpact(selection: any) {
  return fetch(`${API_BASE}/demo/impact`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(selection)
  }).then(res => res.json());
}


export async function getWhatIf(dataset: "my" | "ref", id: string) {
  return fetch(`${API_BASE}/whatif`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, scenario_id: id })
  }).then((res) => {
    if (!res.ok) {
      throw new Error(`API /whatif failed: ${res.status}`);
    }
    return res.json();
  });
}
