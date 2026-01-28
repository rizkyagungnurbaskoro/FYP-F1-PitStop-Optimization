import React from "react";

interface TelemetrySectionProps {
    tireAge: number;
    tireMax: number;
    gapPercentile?: number;
    row: any;
}

export default function TelemetrySection({ tireAge, tireMax, gapPercentile, row }: TelemetrySectionProps) {
    // Calculate tire fill percentage
    const tirePct = Math.min((tireAge / tireMax) * 100, 100);

    // Helpers for display
    const fmt = (val: any, decimals: number = 2) => {
        const num = Number(val);
        return isNaN(num) ? "N/A" : num.toFixed(decimals);
    };

    return (
        <div className="section telemetry-section" style={{ marginTop: 24, paddingTop: 16, borderTop: "1px solid #1e2531" }}>
            <div className="section-title" style={{ fontSize: "1.2rem", borderBottom: "none", marginBottom: 8 }}>
                Telemetry
            </div>

            {/* Tire Gauge */}
            <div style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", color: "#a6adbb", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                    <span>Tire Health / Degradation</span>
                    <span>{row.tire_wear_pct ? (Number(row.tire_wear_pct) * 100).toFixed(1) : (tirePct).toFixed(1)} % Wear ({tireAge.toFixed(0)} Laps)</span>
                </div>
                <div style={{ height: 10, borderRadius: 999, background: "#151c28", border: "1px solid rgba(255,255,255,0.08)", overflow: "hidden", marginTop: 4 }}>
                    <div style={{ height: "100%", width: `${row.tire_wear_pct ? (Number(row.tire_wear_pct) * 100) : tirePct}%`, background: "linear-gradient(90deg, #28c1d6, #ff9d2b, #ff2b2b)" }}></div>
                </div>
                <div style={{ fontSize: "0.65rem", color: "#666", marginTop: 4 }}>
                    Estimated from stint length and thermal degradation models.
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-2" style={{ gap: 12 }}>
                {/* Left Panel: Gap Stats */}
                <div className="card" style={{ padding: 12 }}>
                    <div style={{ fontSize: "0.75rem", color: "#a6adbb", textTransform: "uppercase", marginBottom: 8 }}>PACE / GAPS</div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.85rem" }}>
                        <span style={{ color: "#e6ebf2" }}>Gap to Leader</span>
                        <strong>{fmt(row.gap_to_leader ?? row.gap, 2)}s</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.85rem" }}>
                        <span style={{ color: "#e6ebf2" }}>Gap to Front</span>
                        <strong>{fmt(row.gap_to_front ?? row.gap_to_front_prev, 2)}s</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.85rem" }}>
                        <span style={{ color: "#e6ebf2" }}>Gap to Behind</span>
                        <strong>{fmt(row.gap_to_behind ?? row.gap_to_behind_prev, 2)}s</strong>
                    </div>
                    {gapPercentile !== undefined && gapPercentile !== null && (
                        <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.08)" }}>
                            <div style={{ fontSize: "0.7rem", color: "#a6adbb" }}>GAP PERCENTILE (vs History)</div>
                            <div style={{ height: 6, borderRadius: 999, background: "#151c28", marginTop: 4, position: "relative" }}>
                                <div style={{ position: "absolute", left: `${gapPercentile}%`, top: -3, width: 4, height: 12, background: "#17c3ff", filter: "drop-shadow(0 0 4px #17c3ff)" }}></div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Panel: Track/Weather - Minimal placeholder or extra stats */}
                <div className="card" style={{ padding: 12 }}>
                    <div style={{ fontSize: "0.75rem", color: "#a6adbb", textTransform: "uppercase", marginBottom: 8 }}>ENVIRONMENT</div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.85rem" }}>
                        <span style={{ color: "#e6ebf2" }}>Track Temp</span>
                        <strong>{fmt(row.TrackTemp ?? row.TrackTemp_prev, 1)}°C</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.85rem" }}>
                        <span style={{ color: "#e6ebf2" }}>Air Temp</span>
                        <strong>{fmt(row.AirTemp ?? row.AirTemp_prev, 1)}°C</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.85rem" }}>
                        <span style={{ color: "#e6ebf2" }}>Humidity</span>
                        <strong>{fmt(row.Humidity ?? row.Humidity_prev, 1)}%</strong>
                    </div>
                </div>

                {/* Car Stats (Augmented Data) */}
                <div className="card" style={{ padding: 12 }}>
                    <div style={{ fontSize: "0.75rem", color: "#a6adbb", textTransform: "uppercase", marginBottom: 8 }}>CAR STATUS</div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.85rem" }}>
                        <span style={{ color: "#e6ebf2" }}>Speed</span>
                        <strong>{row.speed !== undefined && row.speed !== null ? fmt(row.speed, 0) : "-"} km/h</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.85rem" }}>
                        <span style={{ color: "#e6ebf2" }}>RPM</span>
                        <strong>{row.rpm !== undefined && row.rpm !== null ? fmt(row.rpm, 0) : "-"}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.85rem" }}>
                        <span style={{ color: "#e6ebf2" }}>Throttle</span>
                        <strong>{row.throttle !== undefined && row.throttle !== null ? fmt(row.throttle, 0) : "-"} %</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.85rem" }}>
                        <span style={{ color: "#e6ebf2" }}>DRS</span>
                        <strong style={{ color: row.drs > 8 ? "#4ade80" : "inherit" }}>{row.drs > 8 ? "OPEN" : "CLOSED"}</strong>
                    </div>
                </div>
            </div>
        </div>
    );
}
