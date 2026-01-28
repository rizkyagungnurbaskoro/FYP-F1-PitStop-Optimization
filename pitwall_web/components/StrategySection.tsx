import React from "react";

interface StrategySectionProps {
    modelCall: string;
    decisionSentence: string;
    impact: number;
}

export default function StrategySection({ modelCall, decisionSentence, impact }: StrategySectionProps) {
    const isPit = (modelCall || "").toUpperCase().includes("PIT") || (modelCall || "").toUpperCase().includes("BOX");
    const impactColor = impact > 0 ? "#38d996" : impact < 0 ? "#ff5c5c" : "#f1c232";
    const impactSign = impact > 0 ? "+" : "";

    return (
        <div className="section" style={{ marginTop: 24 }}>
            <div className="section-title" style={{ fontSize: "1.2rem", borderBottom: 2, marginBottom: 12 }}>STRATEGY CALL</div>

            <div className="grid grid-2" style={{ gap: 12 }}>
                {/* Main Call Card */}
                <div className="card" style={{
                    padding: 16,
                    background: isPit ? "linear-gradient(160deg, rgba(225,6,0,0.15), rgba(16,20,28,0.98))" : "linear-gradient(160deg, rgba(23,195,255,0.08), rgba(16,20,28,0.98))",
                    border: isPit ? "1px solid rgba(225,6,0,0.4)" : "1px solid rgba(23,195,255,0.3)"
                }}>
                    <div style={{ fontSize: "0.85rem", textTransform: "uppercase", color: "#a6adbb", marginBottom: 6 }}>RECOMMENDATION</div>
                    <div style={{ fontFamily: "var(--font-oxanium)", fontSize: "2rem", fontWeight: 700, letterSpacing: "0.05em", color: "#fff" }}>
                        {modelCall}
                    </div>
                    <div style={{ marginTop: 8, fontSize: "0.85rem", color: "#d9e0ea", lineHeight: 1.4 }}>
                        {decisionSentence}
                    </div>
                </div>

                {/* Impact / Reasoning Card */}
                <div className="card" style={{ padding: 16 }}>
                    <div style={{ fontSize: "0.85rem", textTransform: "uppercase", color: "#a6adbb", marginBottom: 6 }}>ESTIMATED IMPACT</div>
                    <div style={{ fontFamily: "var(--font-oxanium)", fontSize: "2rem", fontWeight: 700, color: impactColor }}>
                        {impactSign}{impact.toFixed(2)}s
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "#a6adbb", marginTop: 4 }}>
                        vs Historical Decision
                    </div>
                </div>
            </div>
        </div>
    );
}
