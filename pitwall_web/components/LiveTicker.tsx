"use client";

import React, { useEffect, useState } from "react";

const RACE_EVENTS = [
    { type: "info", icon: "", message: "WET TRACK - STANDING WATER IN SECTOR 3" },
    { type: "warning", icon: "", message: "LOW GRIP - SECTOR 2 TEMPERATURE DROPPING" },
    { type: "pit", icon: "", message: "LAP 16 - PEREZ IN FOR INTERMEDIATES" },
    { type: "info", icon: "", message: "WEATHER UPDATE - DRYING LINE FORMING" },
    { type: "success", icon: "", message: "SECTOR 1 - LECLERC PURPLE ON INTERS" },
    { type: "pit", icon: "", message: "LAP 21 - STRATEGY: BOX FOR SLICKS?" },
    { type: "warning", icon: "", message: "SLIPPERY SURFACE - TURN 1" },
    { type: "info", icon: "", message: "TRACK TEMP: 26.8°C - STABILIZED" },
    { type: "success", icon: "", message: "FASTEST LAP - SAINZ 1:19.421" },
    { type: "info", icon: "", message: "PIT WINDOW OPEN FOR HARD COMPOUND" }
];

export default function LiveTicker() {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [isAnimating, setIsAnimating] = useState(false);

    useEffect(() => {
        const timer = setInterval(() => {
            setIsAnimating(true);
            setTimeout(() => {
                setCurrentIndex((prev) => (prev + 1) % RACE_EVENTS.length);
                setIsAnimating(false);
            }, 300);
        }, 4000);
        return () => clearInterval(timer);
    }, []);

    const currentEvent = RACE_EVENTS[currentIndex];

    const getColorForType = (type: string) => {
        switch (type) {
            case "warning": return "#F1C232";
            case "pit": return "#E10600";
            case "success": return "#38D996";
            default: return "#17C3FF";
        }
    };

    return (
        <div style={{
            marginTop: 16,
            marginBottom: 16,
            display: "flex",
            alignItems: "center",
            gap: 12,
            padding: "10px 20px",
            background: "rgba(16, 20, 28, 0.65)",
            backdropFilter: "blur(12px)",
            WebkitBackdropFilter: "blur(12px)",
            borderRadius: 8,
            border: "1px solid rgba(255,255,255,0.08)",
            boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
            overflow: "hidden",
            position: "relative",
            clipPath: "polygon(0 0, calc(100% - 20px) 0, 100% 50%, calc(100% - 20px) 100%, 0 100%)"
        }}>
            {/* Animated Background */}
            <div style={{
                position: "absolute",
                top: 0,
                left: isAnimating ? "100%" : "-100%",
                width: "100%",
                height: "100%",
                background: "linear-gradient(90deg, transparent, rgba(23, 195, 255, 0.1), transparent)",
                transition: "left 0.6s ease-out"
            }}></div>

            {/* Race Control Badge */}
            <div style={{
                fontFamily: "var(--font-oxanium)",
                fontWeight: 800,
                fontSize: "0.75rem",
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                color: "#FFD700",
                paddingRight: 12,
                borderRight: "2px solid rgba(255, 215, 0, 0.3)",
                whiteSpace: "nowrap"
            }}>
                RACE CONTROL
            </div>

            {/* Icon */}
            <div style={{
                fontSize: "1.2rem",
                filter: "drop-shadow(0 0 4px rgba(255,255,255,0.5))"
            }}>
                {currentEvent.icon}
            </div>

            {/* Message */}
            <div style={{
                flex: 1,
                fontSize: "0.85rem",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "#e9eef7",
                fontWeight: 600,
                fontFamily: "var(--font-oxanium)",
                opacity: isAnimating ? 0 : 1,
                transform: isAnimating ? "translateY(-10px)" : "translateY(0)",
                transition: "all 0.3s ease-out"
            }}>
                {currentEvent.message}
            </div>

            {/* Type Indicator */}
            <div style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: getColorForType(currentEvent.type),
                boxShadow: `0 0 10px ${getColorForType(currentEvent.type)}`,
            }}></div>
        </div>
    );
}
