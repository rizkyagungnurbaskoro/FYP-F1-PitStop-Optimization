"use client";
import React, { useEffect, useState } from "react";

const MESSAGES = [
    { type: "track", icon: "", text: "TRACK CONDITIONS: DAMP / DRYING - TEMP 26.8°C" },
    { type: "info", icon: "", text: "STRATEGY WINDOW: INTERMEDIATE TO SLICKS" },
    { type: "info", icon: "", text: "WEATHER UPDATE: NO FURTHER RAIN EXPECTED" },
    { type: "clear", icon: "OK", text: "RACING LINE DRYING - SECTOR 1/2" },
    { type: "pit", icon: "", text: "PIT OPTION: HARD TIRES READY" },
    { type: "warning", icon: "", text: "HAZARD: STANDING WATER AT TURN 15" }
];

export default function RaceControl() {
    const [idx, setIdx] = useState(0);
    const [isSliding, setIsSliding] = useState(false);

    useEffect(() => {
        const timer = setInterval(() => {
            setIsSliding(true);
            setTimeout(() => {
                setIdx(prev => (prev + 1) % MESSAGES.length);
                setIsSliding(false);
            }, 400);
        }, 4500);
        return () => clearInterval(timer);
    }, []);

    const getColorForType = (type: string) => {
        switch (type) {
            case "warning": return "#F1C232";
            case "pit": return "#E10600";
            case "clear": return "#38D996";
            case "drs": return "#17C3FF";
            default: return "#FFD700";
        }
    };

    const message = MESSAGES[idx];
    const color = getColorForType(message.type);

    return (
        <div style={{
            marginTop: 16, marginBottom: 16,
            display: "flex", alignItems: "center", gap: 14,
            padding: "12px 20px",
            background: "rgba(16, 20, 28, 0.65)",
            backdropFilter: "blur(12px)",
            WebkitBackdropFilter: "blur(12px)",
            borderRadius: 8,
            border: `1px solid ${color}40`,
            boxShadow: `0 8px 32px rgba(0,0,0,0.6), inset 0 0 10px ${color}10`,
            position: "relative",
            overflow: "hidden",
            clipPath: "polygon(0 0, calc(100% - 16px) 0, 100% 50%, calc(100% - 16px) 100%, 0 100%)"
        }}>
            {/* Sliding Background Effect */}
            <div style={{
                position: "absolute",
                top: 0,
                left: isSliding ? "100%" : "-100%",
                width: "100%",
                height: "100%",
                background: `linear-gradient(90deg, transparent, ${color}20, transparent)`,
                transition: "left 0.6s cubic-bezier(0.16, 1, 0.3, 1)"
            }}></div>

            {/* Race Control Badge */}
            <div style={{
                fontFamily: "var(--font-oxanium)", fontWeight: 800, fontSize: "0.8rem",
                textTransform: "uppercase", letterSpacing: "0.12em", color: color,
                paddingRight: 14, borderRight: `2px solid ${color}40`,
                whiteSpace: "nowrap",
                textShadow: `0 0 10px ${color}80`
            }}>
                RACE CONTROL
            </div>

            {/* Icon */}
            <div style={{
                fontSize: "1.1rem",
                display: "flex",
                alignItems: "center",
                filter: "drop-shadow(0 0 4px rgba(255,255,255,0.5))"
            }}>
                {message.icon}
            </div>

            {/* Message */}
            <div style={{
                flex: 1,
                fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.08em",
                color: "#e9eef7", fontWeight: 600,
                fontFamily: "var(--font-oxanium)",
                opacity: isSliding ? 0 : 1,
                transform: isSliding ? "translateX(20px)" : "translateX(0)",
                transition: "all 0.4s cubic-bezier(0.16, 1, 0.3, 1)"
            }}>
                {message.text}
            </div>

            {/* Status Indicator Dot */}
            <div style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: color,
                boxShadow: `0 0 12px ${color}`,
                animation: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite"
            }}></div>
        </div>
    );
}

