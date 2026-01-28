type Context = {
  sc_active?: number | boolean;
  vsc_active?: number | boolean;
  rain?: number | boolean;
  track_temp?: number | null;
  tire_age?: number | null;
  position?: number | null;
  gap?: number | null;
};

export default function ContextChips({ ctx }: { ctx: Context }) {
  const items = [
    `SC ${ctx.sc_active ? "ON" : "OFF"}`,
    `VSC ${ctx.vsc_active ? "ON" : "OFF"}`,
    `Rain ${ctx.rain ? "YES" : "NO"}`,
    `Track ${ctx.track_temp ?? "-"}°C`,
    `Tyre Age ${ctx.tire_age ?? "-"}`,
    `Pos P${ctx.position ?? "-"}`,
    `Gap ${typeof ctx.gap === 'number' ? ctx.gap.toFixed(3) : ctx.gap ?? "-"}s`
  ];

  return (
    <div className="chips" style={{ display: "flex", flexWrap: "wrap", gap: 8, fontFamily: "var(--font-oxanium)" }}>
      {items.map((item) => (
        <span key={item} className="chip" style={{ fontSize: "0.75rem", padding: "4px 8px" }}>
          {item}
        </span>
      ))}
    </div>
  );
}
