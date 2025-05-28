import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Download } from "lucide-react";
import { useNavigate } from "react-router-dom";

const imageOrder = [
  "Original",
  "Preprocessed",
  "Enhanced",
  "Discs Space detection",
];

export default function ImageGrid({ images }) {
  const navigate = useNavigate();
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 justify-center">
      {imageOrder.map((key) =>
        images[key] ? (
          <Card
            key={key}
            className="backdrop-blur-lg bg-white/20 border border-white/30 rounded-2xl shadow-xl hover:scale-105 transition"
          >
            <img
              src={`data:image/png;base64,${images[key]}`}
              alt={key}
              className="rounded-xl w-full h-80 object-cover cursor-pointer transition-transform duration-200 hover:scale-105"
              onClick={() =>
                navigate(`/view/${encodeURIComponent(key)}`, {
                  state: { src: images[key], title: key },
                })
              }
            />
            <CardContent className="text-center">
              <div className="font-semibold text-lg text-gray-900 mt-2">
                {key === "Preprocessed"
                  ? "Preprocessed (CLAHE)"
                  : key === "Enhanced"
                  ? "Enhanced (K-Means)"
                  : key}
              </div>
              <Button variant="outline" className="mt-3 glass-btn" asChild>
                <a
                  href={`data:image/png;base64,${images[key]}`}
                  download={`${key.toLowerCase().replace(/ /g, "_")}.png`}
                >
                  <Download className="mr-2 h-4 w-4" />
                  Download {key} Image
                </a>
              </Button>
            </CardContent>
          </Card>
        ) : null
      )}
    </div>
  );
}
