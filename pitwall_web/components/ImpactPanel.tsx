type ImpactProps = {
  seconds: number;
  note: string;
};

export default function ImpactPanel({ seconds, note }: ImpactProps) {
  return (
    <div className="card">
      <h3>Strategy Impact (Estimated)</h3>
      <div className="value">{seconds > 0 ? "+" : ""}{seconds.toFixed(1)}s</div>
      <div className="notice">{note}</div>
    </div>
  );
}
