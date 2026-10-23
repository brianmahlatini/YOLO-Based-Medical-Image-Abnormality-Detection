export type Detection = {
  label: string;
  confidence: number;
  source: string;
  bbox: {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
  };
};

export type PredictionResponse = {
  disclaimer: string;
  image_size: {
    width: number;
    height: number;
  };
  detections: Detection[];
  annotated_image: string;
  outcome: {
    model_name: string;
    abnormality_probability: number;
    predicted_label: string;
    features: Record<string, number>;
  };
  metrics: {
    dataset?: {
      name: string;
      samples: number;
      train: number;
      validation: number;
      test: number;
      target: string;
    };
    models?: Record<string, unknown>;
    detection?: {
      precision: number;
      recall: number;
      "mAP@50": number;
      "mAP@50-95": number;
    };
  };
};
