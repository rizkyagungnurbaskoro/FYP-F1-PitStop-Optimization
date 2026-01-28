type Context = {
  sc_active?: number | boolean;
  vsc_active?: number | boolean;
  rain?: number | boolean;
  weather_status?: string;
  track_temp?: number | null;
  tire_age?: number | null;
  position?: number | null;
  gap?: number | null;
};

export default function ContextChips({ ctx }: { ctx: Context }) {
  const items = [
    { label: `SC ${ctx.sc_active ? "ON" : "OFF"}`, color: ctx.sc_active ? "#E10600" : "#374151" },
    { label: `VSC ${ctx.vsc_active ? "ON" : "OFF"}`, color: ctx.vsc_active ? "#E10600" : "#374151" },
    {
      label: ctx.weather_status === "Crossover" ? "CROSSOVER" : `Rain ${ctx.rain ? "YES" : "NO"}`,
      color: ctx.weather_status === "Crossover" ? "#f59e0b" : (ctx.rain ? "#3b82f6" : "#374151")
    },
    { label: `Track ${ctx.track_temp ?? "-"}°C`, color: "#10b981" },
    { label: `Tyre Age ${ctx.tire_age ?? "-"}`, color: "#6366f1" },
    { label: `Pos P${ctx.position ?? "-"}`, color: "#f97316" },
    { label: `Gap ${typeof ctx.gap === 'number' ? ctx.gap.toFixed(3) : ctx.gap ?? "-"}s`, color: "#a855f7" }
  ];

  return (
    <div className="chips" style={{ display: "flex", flexWrap: "wrap", gap: 8, fontFamily: "var(--font-oxanium)" }}>
      {items.map((item) => (
        <span key={item.label} className="chip" style={{
          fontSize: "0.75rem",
          padding: "4px 10px",
          background: "rgba(0,0,0,0.4)",
          border: `1px solid ${item.color}`,
          color: item.color,
          borderRadius: 4,
          fontWeight: 800,
          textTransform: "uppercase"
        }}>
          {item.label}
        </span>
      ))}
    </div>
  );
}
