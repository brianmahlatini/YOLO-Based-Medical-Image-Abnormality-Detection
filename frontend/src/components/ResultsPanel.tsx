import { Activity, BrainCircuit, Crosshair, ShieldAlert } from "lucide-react";
import type { PredictionResponse } from "../types";

type ResultsPanelProps = {
  result: PredictionResponse | null;
  previewUrl: string | null;
};

function pct(value: number) {
  return `${Math.round(value * 1000) / 10}%`;
}

export function ResultsPanel({ result, previewUrl }: ResultsPanelProps) {
  if (!previewUrl && !result) {
    return (
      <section className="emptyState">
        <Crosshair aria-hidden="true" />
        <p>Upload an image to view localized regions, confidence scores, and model metrics.</p>
      </section>
    );
  }

  return (
    <section className="resultsGrid">
      <div className="imageStage">
        <img src={result?.annotated_image ?? previewUrl ?? ""} alt="Uploaded medical scan with prediction overlays" />
      </div>

      {result && (
        <aside className="analysisPanel">
          <div className="warning">
            <ShieldAlert aria-hidden="true" />
            <span>{result.disclaimer}</span>
          </div>

          <div className="scoreBlock">
            <div>
              <span className="eyebrow">Runtime model</span>
              <strong>{result.outcome.model_name}</strong>
            </div>
            <div className="probability">{pct(result.outcome.abnormality_probability)}</div>
            <span className="label">{result.outcome.predicted_label.replaceAll("_", " ")}</span>
          </div>

          <div className="metricGrid">
            <div>
              <Activity aria-hidden="true" />
              <span>Precision</span>
              <strong>{pct(result.metrics.detection?.precision ?? 0)}</strong>
            </div>
            <div>
              <BrainCircuit aria-hidden="true" />
              <span>Recall</span>
              <strong>{pct(result.metrics.detection?.recall ?? 0)}</strong>
            </div>
            <div>
              <Crosshair aria-hidden="true" />
              <span>mAP@50</span>
              <strong>{pct(result.metrics.detection?.["mAP@50"] ?? 0)}</strong>
            </div>
            <div>
              <Crosshair aria-hidden="true" />
              <span>mAP@50-95</span>
              <strong>{pct(result.metrics.detection?.["mAP@50-95"] ?? 0)}</strong>
            </div>
          </div>

          <div className="detections">
            <h2>Detected regions</h2>
            {result.detections.length === 0 ? (
              <p>No localized abnormality region exceeded the detector threshold.</p>
            ) : (
              result.detections.map((detection, index) => (
                <div className="detectionRow" key={`${detection.label}-${index}`}>
                  <div>
                    <strong>{detection.label.replaceAll("_", " ")}</strong>
                    <span>{detection.source}</span>
                  </div>
                  <b>{pct(detection.confidence)}</b>
                </div>
              ))
            )}
          </div>
        </aside>
      )}
    </section>
  );
}
