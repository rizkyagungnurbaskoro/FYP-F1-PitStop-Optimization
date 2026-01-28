export default function TimingTower({ rows80, rows70 }: { rows80: any[], rows70: any[] }) {
  // rows80: S1 (Ref), S2 (My), S3 (Ref), S4 (My)
  // Each row has { stage, value, delta? }

  const max80 = rows80.length > 0 ? Math.max(...rows80.map(r => r.value || 0)) : 0;
  const max70 = rows70.length > 0 ? Math.max(...rows70.map(r => r.value || 0)) : 0;

  const RowItem = ({ row, idx, isHoldout = false }: { row: any, idx: number, isHoldout?: boolean }) => {
    const stageLabel = row.stage || `S${idx + 1}`;
    const f1Score = row.value || 0;
    const isMyMethod = row.method === "MyMethod" || (idx % 2 === 1);
    const delta = row.delta ?? null;
    const isBest = f1Score > 0 && Math.abs(f1Score - (isHoldout ? max70 : max80)) < 0.0001;

    // Streamlit colors
    const refColor = "#6e7888";
    const myColor = "#ff2b2b";
    const refGradient = "linear-gradient(90deg, #2c3545, #6e7888)";
    const myGradient = "linear-gradient(90deg, #ff9d2b, #ff2b2b)";

    return (
      <div style={{
        display: "grid",
        gridTemplateColumns: "48px 1fr 70px 70px",
        alignItems: "center",
        gap: 12,
        padding: "8px 12px",
        borderRadius: 12,
        background: "rgba(14, 18, 26, 0.92)",
        border: isBest ? "1px solid rgba(255, 157, 43, 0.7)" : "1px solid rgba(255, 255, 255, 0.08)",
        boxShadow: isBest ? "0 0 12px rgba(255, 157, 43, 0.2)" : "none",
        marginBottom: 8,
        transition: "all 0.2s ease"
      }}>
        {/* Stage */}
        <div style={{
          fontFamily: "var(--font-oxanium)",
          fontWeight: 700,
          color: "#d9e0ea",
          letterSpacing: "0.08em",
          fontSize: "0.85rem"
        }}>
          {stageLabel}
        </div>

        {/* Bar */}
        <div style={{
          height: 8,
          borderRadius: 999,
          background: "#151c28",
          border: "1px solid rgba(255, 255, 255, 0.08)",
          overflow: "hidden",
          position: "relative"
        }}>
          <div style={{
            height: "100%",
            width: `${Math.min(100, f1Score * 100)}%`,
            background: isMyMethod ? myGradient : refGradient,
            borderRadius: 999,
            boxShadow: isMyMethod ? "0 0 10px rgba(255, 43, 43, 0.3)" : "none"
          }} />
        </div>

        {/* Value */}
        <div style={{
          textAlign: "right",
          fontWeight: 700,
          fontSize: "0.85rem",
          color: "#fff",
          fontFamily: "var(--font-rajdhani)"
        }}>
          {f1Score.toFixed(3)}
        </div>

        {/* Delta */}
        <div style={{
          textAlign: "right",
          fontSize: "0.8rem",
          fontWeight: 700,
          color: delta === null ? "#6b7280" : (delta >= 0 ? "#22c55e" : "#ef4444"),
          fontFamily: "var(--font-rajdhani)"
        }}>
          {delta === null ? "N/A" : `${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`}
        </div>
      </div>
    );
  };

  return (
    <div className="card" style={{
      padding: "20px 24px",
      background: "rgba(16, 20, 28, 0.7)",
      backdropFilter: "blur(12px)",
      WebkitBackdropFilter: "blur(12px)",
      borderRadius: 16,
      border: "1px solid rgba(255,255,255,0.08)",
      boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
      fontFamily: "var(--font-oxanium)"
    }}>
      {/* QUICK COMPARISON (implied by context in image) */}
      <div style={{
        fontSize: "1.2rem",
        fontWeight: 900,
        color: "#fff",
        textTransform: "uppercase",
        letterSpacing: "2px",
        marginBottom: 20,
        display: "flex",
        flexDirection: "column",
        gap: 4
      }}>
        QUICK COMPARISON
        <div style={{ height: 3, width: 40, background: "#f97316" }}></div>
      </div>

      {/* TIMING TOWER */}
      <div style={{ marginBottom: 32 }}>
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 12
        }}>
          <h3 style={{
            margin: 0,
            textTransform: "uppercase",
            fontWeight: 700,
            letterSpacing: "0.1em",
            color: "#e6ebf2",
            fontSize: "0.85rem"
          }}>
            TIMING TOWER
          </h3>
          <div style={{ fontSize: "0.7rem", color: "#a6adbb", textTransform: "uppercase", letterSpacing: "0.08em" }}>
            80/20 TRAIN-TEST SPLIT | F1
          </div>
        </div>

        <div>
          {rows80.map((row, idx) => (
            <RowItem key={idx} row={row} idx={idx} />
          ))}
        </div>

        {/* Legend */}
        <div style={{
          marginTop: 12,
          display: "flex",
          gap: 16,
          alignItems: "center",
          fontSize: "0.72rem",
          color: "#9aa4b3",
          textTransform: "uppercase",
          letterSpacing: "0.05em"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#6e7888" }}></span> RefTech
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#ff2b2b" }}></span> MyMethod
          </div>
        </div>
      </div>

      {/* HOLDOUT TOWER */}
      {rows70.length > 0 && (
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 24 }}>
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: 12
          }}>
            <h3 style={{
              margin: 0,
              textTransform: "uppercase",
              fontWeight: 700,
              letterSpacing: "0.1em",
              color: "#e6ebf2",
              fontSize: "0.85rem"
            }}>
              HOLDOUT TOWER
            </h3>
            <div style={{ fontSize: "0.7rem", color: "#a6adbb", textTransform: "uppercase", letterSpacing: "0.08em" }}>
              HOLDOUT 70/30 (STAGE 3/4) | F1
            </div>
          </div>

          <div>
            {rows70.map((row, idx) => (
              <RowItem key={idx} row={row} idx={idx} isHoldout />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

