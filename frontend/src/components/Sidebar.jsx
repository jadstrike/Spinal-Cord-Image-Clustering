import { Button } from "./ui/button";

export default function Sidebar({
  file,
  setFile,
  onAnalyze,
  loading,
  error,
  success,
}) {
  return (
    <aside className="w-full max-w-xs bg-white/30 backdrop-blur-lg border-r border-white/30 flex flex-col items-center py-10 px-6 shadow-xl">
      <h2 className="text-2xl font-extrabold tracking-tight text-gray-900 mb-8 text-center drop-shadow-lg">
        Spinal Cord Image Clustering
      </h2>
      <form onSubmit={onAnalyze} className="w-full flex flex-col gap-4">
        <Button type="button" className="w-full" variant="secondary" asChild>
          <label>
            Upload Spine Image
            <input
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => setFile(e.target.files[0])}
            />
          </label>
        </Button>
        {file && (
          <div className="text-xs text-gray-700 font-medium break-all">
            <b>Selected:</b> {file.name}
          </div>
        )}
        <Button
          type="submit"
          className="w-full font-bold"
          disabled={!file || loading}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </Button>
        {error && (
          <div className="text-red-600 text-sm font-semibold text-center mt-2">
            {error}
          </div>
        )}
        {success && (
          <div className="text-green-600 text-sm font-semibold text-center mt-2">
            Analysis complete!
          </div>
        )}
      </form>
    </aside>
  );
}
