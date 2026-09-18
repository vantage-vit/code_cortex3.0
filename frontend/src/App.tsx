import { Route, Routes } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import ResultsPage from "./pages/ResultsPage";
import HistoryPage from "./pages/HistoryPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<DashboardPage />} />
      <Route path="/results/:fileName" element={<ResultsPage />} />
      <Route path="/history" element={<HistoryPage />} />
    </Routes>
  );
}
