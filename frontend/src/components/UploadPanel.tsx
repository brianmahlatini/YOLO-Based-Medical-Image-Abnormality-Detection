import { UploadCloud } from "lucide-react";
import { ChangeEvent, DragEvent, useRef, useState } from "react";

type UploadPanelProps = {
  onSelect: (file: File) => void;
  disabled?: boolean;
};

export function UploadPanel({ onSelect, disabled }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const pickFile = (file?: File) => {
    if (file && file.type.startsWith("image/")) {
      onSelect(file);
    }
  };

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    pickFile(event.target.files?.[0]);
  };

  const handleDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setIsDragging(false);
    pickFile(event.dataTransfer.files?.[0]);
  };

  return (
    <button
      className={`uploadPanel ${isDragging ? "dragging" : ""}`}
      disabled={disabled}
      onClick={() => inputRef.current?.click()}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      type="button"
    >
      <UploadCloud aria-hidden="true" />
      <span>Upload chest image</span>
      <small>PNG or JPEG image, experimental analysis only</small>
      <input ref={inputRef} type="file" accept="image/*" onChange={handleChange} hidden />
    </button>
  );
}
