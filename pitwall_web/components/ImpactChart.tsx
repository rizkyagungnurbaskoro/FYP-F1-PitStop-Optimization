
import React from 'react';
import { motion } from 'framer-motion';

interface ImpactChartProps {
    aiDecision: string;
    histDecision: string;
    impact: number;
}

export default function ImpactChart({ aiDecision, histDecision, impact }: ImpactChartProps) {
    // If decisions are same, net gain is 0
    const isSame = Math.abs(impact) < 0.1;

    // We want to compare Race Time Delta
    // AI is Baseline (0)
    // History is (+impact if positive, -impact if negative)
    // Actually simplicity is better:
    // Bar 1: AI Strategy (0.00s)
    // Bar 2: Historical Strategy (+X.XXs Slower) or (-X.XXs Faster)

    const gain = impact;
    const isGood = gain >= 0;

    return (
        <div style={{
            background: "rgba(16, 20, 28, 0.45)",
            backdropFilter: "blur(12px)",
            borderRadius: 16,
            padding: "24px",
            border: "1px solid rgba(168, 85, 247, 0.2)",
            fontFamily: "var(--font-oxanium)",
            position: "relative",
            overflow: "hidden"
        }}>
            <div style={{ position: "absolute", top: 0, left: 0, width: 4, height: "100%", background: "#a855f7" }}></div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 15 }}>
                <div>
                    <div style={{ fontSize: "0.75rem", color: "#a855f7", fontWeight: 800, letterSpacing: "2px", textTransform: "uppercase", marginBottom: 4 }}>
                        Time Delta Analysis
                    </div>
                    <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#fff" }}>
                        Race Time Implication
                    </div>
                </div>
                <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: "2rem", fontWeight: 900, color: isGood ? "#22c55e" : "#ef4444", lineHeight: 1 }}>
                        {isSame ? "MATCH" : (isGood ? "-" : "+") + Math.abs(gain).toFixed(2) + "s"}
                    </div>
                    <div style={{ fontSize: "0.7rem", color: "#9ca3af", marginTop: 4 }}>
                        {isSame ? "RECOMMENDATION MATCHES REALITY" : (isGood ? "FASTER THAN HISTORY" : "SLOWER THAN HISTORY")}
                    </div>
                </div>
            </div>

            <div style={{ display: "grid", gridTemplateRows: "1fr 1fr", gap: 12 }}>
                {/* AI BAR (Baseline) */}
                <div style={{ display: "grid", gridTemplateColumns: "100px 1fr 60px", alignItems: "center", gap: 10 }}>
                    <div style={{ fontSize: "0.8rem", fontWeight: 800, color: "#E10600", textAlign: "right" }}>AI STRATEGY</div>
                    <div style={{ height: 32, background: "rgba(255,255,255,0.05)", borderRadius: 4, position: "relative" }}>
                        <div style={{
                            position: "absolute",
                            top: 0, bottom: 0, left: 0,
                            width: "70%",
                            background: "linear-gradient(90deg, #E10600, #ff4d4d)",
                            borderRadius: 4,
                            opacity: 0.8
                        }}></div>
                    </div>
                    <div style={{ fontSize: "0.9rem", fontWeight: 900, color: "#fff" }}>BASE</div>
                </div>

                {/* HISTORICAL BAR */}
                <div style={{ display: "grid", gridTemplateColumns: "100px 1fr 60px", alignItems: "center", gap: 10 }}>
                    <div style={{ fontSize: "0.8rem", fontWeight: 800, color: "#17C3FF", textAlign: "right" }}>HISTORY</div>
                    <div style={{ height: 32, background: "rgba(255,255,255,0.05)", borderRadius: 4, position: "relative" }}>
                        {/* 
                           If AI is Faster (Gain > 0), History Bar should be LONGER (slower time).
                           If AI is Slower (Gain < 0), History Bar should be SHORTER (faster time).
                        */}
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: isSame ? "70%" : (isGood ? "90%" : "50%") }}
                            transition={{ duration: 0.8, ease: "circOut" }}
                            style={{
                                position: "absolute",
                                top: 0, bottom: 0, left: 0,
                                background: isSame ? "#17C3FF" : (isGood ? "rgba(255,255,255,0.2)" : "#22c55e"),
                                borderRadius: 4
                            }}>
                            {!isSame && (
                                <div style={{
                                    position: "absolute",
                                    right: 0, top: 0, bottom: 0, width: isGood ? "22%" : 0,
                                    background: "#ef4444",
                                    opacity: isGood ? 1 : 0
                                }}></div>
                            )}
                        </motion.div>
                    </div>
                    <div style={{ fontSize: "0.9rem", fontWeight: 900, color: isGood ? "#ef4444" : "#22c55e" }}>
                        {isSame ? "SAME" : (isGood ? "+" + Math.abs(gain).toFixed(1) + "s" : "-" + Math.abs(gain).toFixed(1) + "s")}
                    </div>
                </div>
            </div>

            <div style={{ marginTop: 12, fontSize: "0.7rem", color: "#666", textAlign: "center", fontStyle: "italic" }}>
                * Projected race time delta calculated from degradation trajectory
            </div>
        </div>
    );
}
