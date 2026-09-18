import { Route, Routes } from "react-router-dom";
import AnalyzePage from "./pages/AnalyzePage";
import ResultsPage from "./pages/ResultsPage";
import HistoryPage from "./pages/HistoryPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<AnalyzePage />} />
      <Route path="/results/:fileName" element={<ResultsPage />} />
      <Route path="/history" element={<HistoryPage />} />
    </Routes>
  );
}
