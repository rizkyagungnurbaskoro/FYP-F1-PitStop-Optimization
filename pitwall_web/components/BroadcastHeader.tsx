"use client";

import React, { useEffect, useState } from "react";

export default function BroadcastHeader({ circuit }: { circuit?: string }) {
    const [currentTime, setCurrentTime] = useState(new Date());
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        const timer = setInterval(() => {
            setCurrentTime(new Date());
        }, 1000);
        return () => clearInterval(timer);
    }, []);

    const circuitLabel = circuit ? circuit.toUpperCase() + " GRAND PRIX" : "PITWALL STRATEGY";

    return (
        <div className="broadcast-header" style={{
            background: "rgba(16, 20, 28, 0.6)",
            backdropFilter: "blur(16px)",
            WebkitBackdropFilter: "blur(16px)",
            borderBottom: "3px solid #E10600",
            padding: "16px 24px",
            position: "relative",
            overflow: "hidden",
            marginBottom: 24,
            boxShadow: "0 4px 30px rgba(0, 0, 0, 0.5)"
        }}>
            {/* Animated Top Border */}
            <div style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: 3,
                background: "linear-gradient(90deg, #E10600 0%, #17C3FF 50%, #FFD700 100%)",
                backgroundSize: "200% 100%",
                animation: "shimmer 3s linear infinite"
            }}></div>

            <div style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
                gap: 16
            }}>
                {/* Left: F1 Logo + Race Name */}
                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                    <div style={{
                        width: 50,
                        height: 50,
                        background: "#E10600",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontFamily: "var(--font-oxanium)",
                        fontWeight: 900,
                        fontSize: "1.4rem",
                        color: "#fff",
                        borderRadius: 6,
                        boxShadow: "0 0 20px rgba(225, 6, 0, 0.6)",
                        clipPath: "polygon(10% 0%, 100% 0%, 100% 100%, 0% 100%)"
                    }}>
                        F1
                    </div>

                    <div>
                        <div style={{
                            fontFamily: "var(--font-oxanium)",
                            fontSize: "1.8rem",
                            fontWeight: 800,
                            letterSpacing: "0.08em",
                            textTransform: "uppercase",
                            lineHeight: 1
                        }}>
                            {circuitLabel}
                        </div>
                        <div style={{
                            fontSize: "0.75rem",
                            color: "#17C3FF",
                            textTransform: "uppercase",
                            letterSpacing: "0.15em",
                            fontWeight: 600,
                            marginTop: 4,
                            fontFamily: "var(--font-oxanium)"
                        }}>
                            {circuit ? "RACE STRATEGY OPTIMIZATION" : "every seconds matters"}
                        </div>
                    </div>
                </div>

                {/* Right: Live Indicator + Time */}
                <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
                    {/* Live Indicator */}
                    <div className="live-indicator" style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "6px 14px",
                        background: "rgba(225, 6, 0, 0.2)",
                        border: "2px solid #E10600",
                        borderRadius: 6,
                        fontFamily: "var(--font-oxanium)",
                        fontSize: "0.85rem",
                        fontWeight: 800,
                        textTransform: "uppercase",
                        letterSpacing: "0.1em"
                    }}>
                        <div className="live-dot" style={{
                            width: 10,
                            height: 10,
                            background: "#E10600",
                            borderRadius: "50%",
                            boxShadow: "0 0 10px #E10600",
                            animation: "pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite"
                        }}></div>
                        LIVE
                    </div>

                    {/* Clock */}
                    <div style={{
                        fontFamily: "var(--font-oxanium)",
                        fontSize: "1.2rem",
                        fontWeight: 700,
                        color: "#a6adbb",
                        letterSpacing: "0.05em"
                    }} suppressHydrationWarning>
                        {mounted ? currentTime.toLocaleTimeString('en-US', { hour12: false }) : '--:--:--'}
                    </div>
                </div>
            </div>
        </div>
    );
}
