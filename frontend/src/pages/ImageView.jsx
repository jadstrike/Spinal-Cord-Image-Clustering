import { useLocation, useNavigate, useParams } from "react-router-dom";
import { Button } from "../components/ui/button";
import { ArrowLeft, Download } from "lucide-react";

export default function ImageView() {
  const navigate = useNavigate();
  const { state } = useLocation();
  const { type } = useParams();

  if (!state?.src) {
    // If user navigates directly, go back
    navigate("/");
    return null;
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-[#e0eafc] to-[#cfdef3]">
      <div className="w-full flex justify-between items-center mb-8 px-8">
        <Button variant="ghost" onClick={() => navigate(-1)}>
          <ArrowLeft className="mr-2" /> Back
        </Button>
        <Button asChild variant="outline">
          <a
            href={`data:image/png;base64,${state.src}`}
            download={`${type.toLowerCase().replace(/ /g, "_")}.png`}
          >
            <Download className="mr-2" /> Download
          </a>
        </Button>
      </div>
      <img
        src={`data:image/png;base64,${state.src}`}
        alt={type}
        className="max-w-4xl w-full max-h-[80vh] rounded-2xl shadow-2xl border border-white/40 bg-white/30 backdrop-blur-lg"
      />
      <div className="mt-6 text-2xl font-bold text-gray-900">{type}</div>
    </div>
  );
}
