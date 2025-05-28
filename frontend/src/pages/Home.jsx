import { useState } from "react";
import Sidebar from "../components/Sidebar";
import ImageGrid from "../components/ImageGrid";

export default function Home() {
  const [file, setFile] = useState(null);
  const [images, setImages] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError("");
    setImages({});
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/analyze", { method: "POST", body: formData });
      const data = await res.json();
      setImages({
        Original: data.original,
        Preprocessed: data.preprocessed,
        Enhanced: data.enhanced,
        "Discs Space detection": data.discs_space_detection,
      });
      setSuccess(true);
    } catch (err) {
      setError("Failed to analyze image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-[#e0eafc] to-[#cfdef3]">
      <Sidebar
        file={file}
        setFile={setFile}
        onAnalyze={handleAnalyze}
        loading={loading}
        error={error}
        success={success}
      />
      <main className="flex-1 p-8 flex flex-col items-center">
        <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-gray-900 mb-10 text-center drop-shadow-lg">
          Spinal Cord Image Clustering and Analysis
        </h1>
        <ImageGrid images={images} />
      </main>
    </div>
  );
}
