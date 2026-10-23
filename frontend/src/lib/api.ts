import type { PredictionResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export async function predictImage(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail ?? "Prediction request failed.");
  }

  return response.json();
}
