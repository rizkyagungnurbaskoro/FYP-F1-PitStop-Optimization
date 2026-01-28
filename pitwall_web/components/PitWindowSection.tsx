import React from "react";

interface PitWindowSectionProps {
    start?: number;
    end?: number;
    currentLap: number;
    text: string;
}

export default function PitWindowSection({ start, end, currentLap, text }: PitWindowSectionProps) {
    if (!start || !end) {
        return (
            <div className="section" style={{ marginTop: 24 }}>
                <div className="card" style={{ padding: 12, textAlign: "center", color: "#a6adbb" }}>
                    NO PIT WINDOW DATA
                </div>
            </div>
        )
    }

    const range = end - start;
    const progress = currentLap - start;
    const pct = Math.min(Math.max((progress / range) * 100, 0), 100);
    const inWindow = currentLap >= start && currentLap <= end;

    return (
        <div className="section" style={{ marginTop: 24, fontFamily: "var(--font-oxanium)" }}>
            <div className="section-title" style={{ fontSize: "1.2rem", borderBottom: "none", marginBottom: 8 }}>
                Pit Window {text ? `(${text})` : ""}
            </div>
            <div className="card" style={{ padding: "16px 20px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, fontSize: "0.85rem", color: "#a6adbb", fontFamily: "var(--font-oxanium)" }}>
                    <span>OPEN: L{start}</span>
                    <span>CLOSE: L{end}</span>
                </div>

                {/* Timeline Bar */}
                <div style={{ height: 12, borderRadius: 999, background: "#1c2330", border: "1px solid #2a3342", position: "relative", overflow: "hidden" }}>
                    {/* Window Zone */}
                    <div style={{ position: "absolute", left: 0, width: "100%", height: "100%", background: "linear-gradient(90deg, rgba(255,157,43,0.1), rgba(255,157,43,0.3))" }}></div>

                    {/* Current Lap Marker */}
                    {/* We can't really map current lap exactly if it's outside window visually without scaling, 
                        so let's assume bar represents window duration + buffer?
                        For simplicity, let's just show progress WITHIN window if active.
                    */}
                    <div style={{
                        position: "absolute",
                        left: `${pct}%`,
                        top: 0,
                        width: 4,
                        height: "100%",
                        background: inWindow ? "#2bd97f" : "#a6adbb",
                        boxShadow: inWindow ? "0 0 8px #2bd97f" : "none"
                    }}></div>
                </div>

                <div style={{ marginTop: 8, textAlign: "center", fontSize: "0.9rem", fontWeight: 700, color: inWindow ? "#2bd97f" : "#a6adbb" }}>
                    {inWindow ? "WINDOW OPEN" : currentLap < start ? `OPENS IN ${start - currentLap} LAPS` : "WINDOW CLOSED"}
                </div>
            </div>
        </div>
    );
}
