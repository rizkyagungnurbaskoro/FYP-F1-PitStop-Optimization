import TopBar from "../../components/TopBar";
import OverviewClient from "../../components/OverviewClient";

export default function OverviewPage() {
  return (
    <div style={{ minHeight: "100vh", background: "#0c0e12" }}>
      <TopBar />
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "40px 20px" }}>
        <OverviewClient />
      </div>
    </div>
  );
}
