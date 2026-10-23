import { useMemo, useState } from "react";
import { ResultsPanel } from "./components/ResultsPanel";
import { UploadPanel } from "./components/UploadPanel";
import { predictImage } from "./lib/api";
import type { PredictionResponse } from "./types";
import "./styles.css";

export default function App() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const datasetSummary = useMemo(() => {
    const dataset = result?.metrics.dataset;
    if (!dataset) {
      return "Synthetic research-style dataset with YOLO-format bounding boxes";
    }
    return `${dataset.samples} images | train ${dataset.train} | validation ${dataset.validation} | test ${dataset.test}`;
  }, [result]);

  async function handleSelect(file: File) {
    setLoading(true);
    setError(null);
    setResult(null);
    setPreviewUrl(URL.createObjectURL(file));
    try {
      setResult(await predictImage(file));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header className="topBar">
        <div>
          <span className="eyebrow">Computer Vision + Deep Learning</span>
          <h1>YOLO-Based Medical Image Abnormality Detection</h1>
        </div>
        <div className="datasetPill">{datasetSummary}</div>
      </header>

      <section className="workspace">
        <UploadPanel onSelect={handleSelect} disabled={loading} />
        {loading && <div className="status">Running inference...</div>}
        {error && <div className="error">{error}</div>}
        <ResultsPanel result={result} previewUrl={previewUrl} />
      </section>
    </main>
  );
}
