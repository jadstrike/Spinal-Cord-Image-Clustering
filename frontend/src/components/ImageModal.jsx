import { Dialog, DialogContent } from "./ui/dialog";
import { X } from "lucide-react";

export default function ImageModal({ image, onClose }) {
  return (
    <Dialog
      open={!!image}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-w-3xl bg-white/30 backdrop-blur-lg border border-white/40 rounded-2xl p-0 overflow-hidden relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 bg-white/60 hover:bg-white/80 rounded-full p-2 shadow-md"
          aria-label="Close"
        >
          <X className="w-7 h-7 text-gray-700" />
        </button>
        {image && (
          <div className="flex flex-col items-center">
            <img
              src={image.src}
              alt={image.title}
              className="max-w-[80vw] max-h-[75vh] rounded-xl mb-4 mt-2 shadow-lg"
            />
            <div className="font-bold text-lg text-gray-900 mb-2 text-center">
              {image.title}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
