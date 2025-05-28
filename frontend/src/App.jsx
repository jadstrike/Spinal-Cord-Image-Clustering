import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import ImageView from "./pages/ImageView";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/view/:type" element={<ImageView />} />
      </Routes>
    </BrowserRouter>
  );
}
