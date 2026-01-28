type CallPanelProps = {
  recommendation: string;
  prob: number;
  threshold: number;
};

export default function CallPanel({ recommendation, prob, threshold }: CallPanelProps) {
  const key = (recommendation || "").toLowerCase();
  const isPit = key.includes("pit") || key.includes("box");
  const isStandby = key.includes("standby");
  const pillClass = isPit ? "call-pill pit" : isStandby ? "call-pill standby" : "call-pill stayout";

  return (
    <div className="card">
      <h3>Pitwall Call</h3>
      <div className="call-panel">
        <span className={pillClass}>{recommendation}</span>
        <div>
          <div>P(pit) {(prob ?? 0).toFixed(2)}</div>
          <div>Threshold {(threshold ?? 0).toFixed(2)}</div>
        </div>
      </div>
    </div>
  );
}
