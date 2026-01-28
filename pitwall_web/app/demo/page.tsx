import TopBar from "../../components/TopBar";
import DemoClient from "../../components/DemoClient";

export default function DemoPage() {
  return (
    <div className="shell">
      <TopBar />
      <div className="section">
        <div className="section-title">Strategy Demo</div>
        <DemoClient />
      </div>
    </div>
  );
}
