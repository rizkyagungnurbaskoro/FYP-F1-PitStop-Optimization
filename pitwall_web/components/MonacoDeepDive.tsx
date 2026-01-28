"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";

// Import Monaco3DTrack with no SSR to avoid Three.js server-side errors
const Monaco3DTrack = dynamic(() => import("./Monaco3DTrackClient"), {
    ssr: false,
    loading: () => (
        <div style={{
            width: '100%',
            height: '600px',
            background: '#0a0a0c',
            borderRadius: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#17c3ff',
            fontFamily: 'var(--font-oxanium)',
            fontSize: '1.2rem',
            fontWeight: 700
        }}>
            Loading 3D View...
        </div>
    )
});

export default function MonacoDeepDive() {
    const [data, setData] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [viewMode, setViewMode] = useState<"2D" | "3D">("2D");

    useEffect(() => {
        setLoading(true);
        const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
        fetch(`${apiBase}/demo/iconic`)
            .then(res => res.json())
            .then(json => {
                const rawScenarios = json.scenarios || [];
                const filtered = rawScenarios.filter((s: any) => s.race_id === "2022_Monaco");
                // --- CUSTOM LOGO MAPPING: CHANGE YOUR IMAGE URLs HERE ---
                const mappedData = filtered.map((driver: any) => {
                    if (driver.driver === "LEC") {
                        return { ...driver, logo_url: "/img/ferrari.png" }; // Paste Leclerc / Ferrari logo URL here
                    }
                    if (driver.driver === "SAI") {
                        return { ...driver, logo_url: "/img/ferrari.png" }; // Paste Sainz logo URL here
                    }
                    return driver;
                });
                setData(mappedData);
            })
            .catch(err => console.error(err))
            .finally(() => setLoading(false));
    }, []);

    const [hoverX, setHoverX] = useState<number | null>(null);

    if (loading) return <div className="section"><div className="card">Loading Monaco Data...</div></div>;
    if (!data.length) return null;

    // Speed Data Generation
    const zones = [
        { label: "1", x: 0, w: 45, speed: 65, width: 35, type: "LOW" },
        { label: "2", x: 60, w: 45, speed: 110, width: 35, type: "LOW" },
        { label: "3", x: 220, w: 55, speed: 135, width: 45, type: "MED" },
        { label: "4", x: 380, w: 45, speed: 105, width: 35, type: "LOW" },
        { label: "5", x: 440, w: 45, speed: 60, width: 35, type: "LOW" },
        { label: "6", x: 600, w: 45, speed: 50, width: 30, type: "HIGH" },
        { label: "7", x: 660, w: 45, speed: 55, width: 30, type: "HIGH" },
        { label: "8", x: 840, w: 50, speed: 85, width: 40, type: "MED" },
        { label: "9", x: 900, w: 50, speed: 90, width: 40, type: "MED" },
        { label: "10", x: 1050, w: 55, speed: 120, width: 45, type: "MED" },
        { label: "11", x: 1180, w: 20, speed: 80, width: 50, type: "HIGH" }
    ];

    const generateSpeedProfile = (paceDelta: number, isLec: boolean) => {
        const points = 1200;
        const profile = [];
        for (let i = 0; i <= points; i++) {
            const x = (i / points) * 1200;
            let invSum = 0;
            zones.forEach(z => {
                const cx = z.x + (z.w || 0) / 2;
                const dx = x - cx;
                let v = 305;
                if (dx < 0) v = z.speed + (305 - z.speed) * (1 - Math.pow(Math.max(0, 1 + dx / 110), 3));
                else v = z.speed + (310 - z.speed) * (1 - Math.exp(-dx / 55));
                invSum += 1 / (Math.pow(v, 8) + 1);
            });
            const finalBase = Math.pow(1 / (invSum / zones.length), 1 / 8);
            let edgeBuffer = 0;
            if (x < 120) edgeBuffer = 300 * (1 - x / 120);
            if (x > 1180) edgeBuffer = 300 * ((x - 1180) / 20);
            const driverBias = isLec ? (x < 600 ? 5 : -4) : (x < 600 ? -4 : 7);
            const noise = Math.sin(i * 0.1) * (isLec ? 0.9 : -0.9);
            profile.push({ x, speed: Math.max(edgeBuffer, finalBase) + (paceDelta * -8) + driverBias + noise });
        }
        return profile;
    };

    const d1 = data.find(d => d.driver === "LEC") || data[0];
    const d2 = data.find(d => d.driver === "SAI") || data[1];
    const d1Data = data.length > 1 ? generateSpeedProfile(d1.telemetry?.pace_delta || 0, true) : [];
    const d2Data = data.length > 1 ? generateSpeedProfile(d2.telemetry?.pace_delta || 0.1, false) : [];

    // Track Map Component (Central)
    const MonacoMap = ({ drivers, d1Data, d2Data, hoverX }: { drivers: any[], d1Data: any[], d2Data: any[], hoverX: number | null }) => {
        const pathS1 = "M 1.2 670.5 L 0.0 613.3 L 4.1 568.6 L 10.4 533.4 L 19.3 499.1 L 28.1 476.3 L 43.6 464.7 L 71.2 459.8 L 98.1 457.5 L 138.0 451.5 L 169.3 444.2 L 208.1 434.5 L 241.7 425.9 L 283.1 419.2 L 322.0 413.6 L 374.7 389.4 L 420.3 373.8 L 475.1 365.9 L 511.6 357.8 L 549.7 333.0 L 564.0 302.1 L 564.2 273.5 L 556.4 249.8 L 534.9 218.6 L 517.8 194.1 L 516.1 170.0 L 525.2 148.5 L 538.7 129.3 L 564.4 100.5 L 585.2 71.8 L 604.7 44.2 L 624.1 20.5";
        const pathS2 = "M 624.1 20.5 L 636.4 7.4 L 655.7 1.0 L 667.6 8.7 L 673.2 19.6 L 678.2 38.4 L 685.4 59.4 L 698.0 83.6 L 707.7 94.0 L 720.5 97.7 L 725.6 90.8 L 723.5 80.0 L 721.3 75.6 L 711.0 63.1 L 701.2 52.7 L 695.0 41.5 L 695.5 29.9 L 705.8 17.2 L 721.9 8.9 L 742.5 2.0 L 759.7 0.0 L 775.4 7.9 L 779.4 22.0 L 779.9 42.0 L 778.5 62.8 L 776.0 91.5 L 774.5 117.4 L 769.8 158.7 L 759.1 197.6 L 749.9 222.9 L 727.3 274.2 L 706.7 305.4 L 663.6 340.0 L 610.5 365.8 L 583.2 379.7 L 513.0 411.3 L 485.8 420.4 L 436.3 430.5 L 404.1 434.1 L 375.1 438.6 L 363.5 449.9 L 352.5 463.3 L 325.1 463.0 L 296.6 461.1 L 276.6 464.2 L 241.6 469.1 L 209.0 473.0 L 177.2 476.5 L 134.1 479.9 L 103.9 483.7 L 77.8 505.4 L 63.3 528.8 L 54.1 555.0";
        const pathS3 = "M 54.1 555.0 L 48.9 584.3 L 48.0 623.8 L 66.3 659.3 L 82.2 697.3 L 92.5 742.3 L 96.6 778.3 L 92.0 811.4 L 81.6 825.3 L 82.8 848.4 L 91.1 876.4 L 107.1 906.7 L 126.0 929.2 L 147.5 947.4 L 161.3 957.2 L 174.4 967.1 L 182.5 979.4 L 175.9 987.8 L 164.4 994.4 L 151.7 997.6 L 133.8 1000.0 L 109.2 998.8 L 97.2 991.3 L 92.3 980.1 L 88.4 962.4 L 76.4 935.9 L 61.3 913.7 L 45.6 882.7 L 34.5 850.0 L 17.2 788.2 L 10.8 761.0 L 4.2 713.2 L 1.1 668.7 L 1.2 670.5 L 0.0 613.3";

        // Generate Micro-Sector Segments
        const totalPath = `${pathS1} ${pathS2} ${pathS3}`;
        const rawSegments = totalPath.split(" L ");
        const s1Len = pathS1.split(" L ").length;
        const s2Len = pathS2.split(" L ").length;

        // Ghost Driver Logic
        const allPoints = rawSegments.map(p => {
            const parts = p.replace("M ", "").trim().split(" ");
            return { x: Number(parts[0]), y: Number(parts[parts.length - 1]) };
        });

        const getGhostCoords = (val: number | null) => {
            if (val === null) return null;
            const idx = (val / 1200) * (allPoints.length - 1);
            const low = Math.floor(idx);
            const high = Math.min(allPoints.length - 1, Math.ceil(idx));
            const t = idx - low;
            return {
                x: allPoints[low].x + (allPoints[high].x - allPoints[low].x) * t,
                y: allPoints[low].y + (allPoints[high].y - allPoints[low].y) * t
            };
        };
        const ghost = getGhostCoords(hoverX);

        // Sector Purple Logic
        const calcSectorSpeed = (data: any[], start: number, end: number) => {
            const startIdx = Math.floor(start / rawSegments.length * 1200);
            const endIdx = Math.floor(end / rawSegments.length * 1200);
            const slice = data.slice(startIdx, endIdx);
            if (slice.length === 0) return 0;
            return slice.reduce((acc, curr) => acc + curr.speed, 0) / slice.length;
        };

        const lecS1 = calcSectorSpeed(d1Data, 0, s1Len);
        const saiS1 = calcSectorSpeed(d2Data, 0, s1Len);
        const lecS2 = calcSectorSpeed(d1Data, s1Len, s1Len + s2Len);
        const saiS2 = calcSectorSpeed(d2Data, s1Len, s1Len + s2Len);
        const lecS3 = calcSectorSpeed(d1Data, s1Len + s2Len, rawSegments.length);
        const saiS3 = calcSectorSpeed(d2Data, s1Len + s2Len, rawSegments.length);

        const isPurple = (lec: number, sai: number) => Math.abs(lec - sai) > 12; // High performance threshold

        const microSectors = rawSegments.map((s, i) => {
            if (i === 0) return null;
            const prev = rawSegments[i - 1].split(" ").slice(-2);
            const curr = s.split(" ").slice(-2);
            const xVal = (i / rawSegments.length) * 1200;
            const isLecFaster = d1Data[Math.floor(xVal)]?.speed > d2Data[Math.floor(xVal)]?.speed;

            let color = isLecFaster ? "#dc2626" : "#fbbf24";
            if (i < s1Len && isPurple(lecS1, saiS1)) color = "#a855f7";
            if (i >= s1Len && i < s1Len + s2Len && isPurple(lecS2, saiS2)) color = "#a855f7";
            if (i >= s1Len + s2Len && isPurple(lecS3, saiS3)) color = "#a855f7";

            return { d: `M ${prev[0]} ${prev[1]} L ${curr[0]} ${curr[1]}`, color };
        }).filter((seg): seg is { d: string; color: string } => seg !== null);

        // Turns
        // Turns - accurately positioned on the SVG track path
        const turns = [
            { x: 30, y: 640, num: "1" },
            { x: 130, y: 460, num: "2" },
            { x: 310, y: 415, num: "3" },
            { x: 440, y: 395, num: "4" },
            { x: 560, y: 320, num: "5" },
            { x: 670, y: 35, num: "6" },
            { x: 740, y: 60, num: "7" },
            { x: 775, y: 150, num: "8" },
            { x: 680, y: 280, num: "9" },
            { x: 500, y: 420, num: "10" },
            { x: 350, y: 450, num: "11" },
            { x: 170, y: 490, num: "12" },
            { x: 85, y: 550, num: "13" },
            { x: 65, y: 640, num: "14" },
            { x: 90, y: 760, num: "15" },
            { x: 140, y: 840, num: "16" },
            { x: 120, y: 990, num: "18" },
            { x: 25, y: 750, num: "19" }
        ];

        return (
            <div style={{ height: "100%", display: "flex", justifyContent: "center", alignItems: "center", position: "relative" }}>
                {/* Track Map SVG */}
                <svg viewBox="-100 -150 1100 1450" style={{ height: "600px", overflow: "visible" }}>
                    <defs>
                        <filter id="mapGlow" x="-50%" y="-50%" width="200%" height="200%">
                            <feGaussianBlur stdDeviation="8" result="blur" />
                            <feComposite in="SourceGraphic" in2="blur" operator="over" />
                        </filter>
                    </defs>

                    {/* Shadow Layer */}
                    <path d={`${pathS1} ${pathS2} ${pathS3}`} fill="none" stroke="#000" strokeWidth="32" strokeLinecap="round" opacity="0.4" />

                    {/* Micro-Sectors (Dynamic Logic) */}
                    {microSectors.map((s, i) => {
                        let color = s.color;
                        if (i < s1Len && isPurple(lecS1, saiS1)) color = "#a855f7";
                        if (i >= s1Len && i < s1Len + s2Len && isPurple(lecS2, saiS2)) color = "#a855f7";
                        if (i >= s1Len + s2Len && isPurple(lecS3, saiS3)) color = "#a855f7";
                        return <path key={i} d={s.d} fill="none" stroke={color} strokeWidth="24" strokeLinecap="round" />;
                    })}

                    {/* Ghost Driver */}
                    {ghost && (
                        <g filter="url(#mapGlow)">
                            <circle cx={ghost.x} cy={ghost.y} r="18" fill="#fff" opacity="0.8" />
                            <circle cx={ghost.x} cy={ghost.y} r="8" fill="#17c3ff" />
                        </g>
                    )}

                    {/* Glowing Racing Line Overlay */}
                    <path d={`${pathS1} ${pathS2} ${pathS3}`} fill="none" stroke="#fff" strokeWidth="4" strokeLinecap="round" opacity="0.3" filter="url(#mapGlow)" />

                    {/* Grouped Labels with Leader Lines (Massive Offset) */}
                    {[
                        { label: "LOW SPEED", x: -80, y: 640, tx: 30, ty: 640, color: "#fff" },
                        { label: "MED SPEED", x: 310, y: 250, tx: 310, ty: 415, color: "#fbbf24" },
                        { label: "LOW SPEED", x: 700, y: -100, tx: 670, ty: 35, color: "#fff" },
                        { label: "LOW SPEED", x: 500, y: 550, tx: 500, ty: 420, color: "#fff" },
                        { label: "MED SPEED", x: 250, y: 560, tx: 170, ty: 490, color: "#fbbf24" },
                        { label: "MED SPEED", x: 300, y: 920, tx: 140, ty: 840, color: "#fbbf24" },
                        { label: "LOW SPEED", x: -10, y: 1050, tx: 120, ty: 990, color: "#fff" }
                    ].map((g, i) => (
                        <g key={i}>
                            <line x1={g.x} y1={g.y} x2={g.tx} y2={g.ty} stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" strokeDasharray="4 4" />
                            <g transform={`translate(${g.x}, ${g.y})`}>
                                <text y="-20" fill={g.color} fontSize="20" fontWeight="900" textAnchor="middle" fontFamily="var(--font-oxanium)" style={{ letterSpacing: "2.5px" }}>{g.label}</text>
                                <circle r="6" fill="#fff" />
                            </g>
                        </g>
                    ))}

                    {/* High-Visibility Enlarged Turn Numbers */}
                    {turns.map((t, i) => (
                        <g key={i}>
                            <text
                                x={t.x}
                                y={t.y + 32}
                                fill="#fff"
                                fontSize="24"
                                fontWeight="900"
                                textAnchor="middle"
                                fontFamily="var(--font-oxanium)"
                                style={{ filter: "drop-shadow(0px 0px 8px rgba(0,0,0,1))" }}
                            >
                                {t.num}
                            </text>
                        </g>
                    ))}
                </svg>
            </div>
        );
    };

    // Compound Icon helper
    const CompoundIcon = ({ compound }: { compound: string }) => {
        const c = compound?.toUpperCase() || "INTER";
        let color = "#fff";
        let border = "#fff";
        let label = "I";

        if (c.includes("SOFT")) { color = "#dc2626"; label = "S"; }
        else if (c.includes("MEDIUM")) { color = "#facc15"; label = "M"; }
        else if (c.includes("HARD")) { color = "#fff"; label = "H"; }
        else if (c.includes("INTER")) { color = "#22c55e"; label = "I"; }
        else if (c.includes("WET")) { color = "#3b82f6"; label = "W"; }

        return (
            <div style={{
                width: 28, height: 28,
                borderRadius: "50%",
                border: `2px solid ${color}`,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: "1rem", fontWeight: 900, color: color,
                fontFamily: "var(--font-oxanium)"
            }} title={c}>
                {label}
            </div>
        );
    };

    // Track Atmospherics Widget
    const TrackConditions = () => (
        <div style={{
            display: "flex", gap: 30, padding: "12px 24px",
            background: "rgba(16, 20, 28, 0.6)",
            backdropFilter: "blur(12px)",
            borderRadius: 12, border: "1px solid rgba(23, 195, 255, 0.2)",
            boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
            fontFamily: "var(--font-oxanium)"
        }}>
            <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "0.6rem", color: "#9ca3af", fontWeight: 800 }}>TRACK TEMP</div>
                <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "#17c3ff" }}>28.8°C</div>
            </div>
            <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "0.6rem", color: "#9ca3af", fontWeight: 800 }}>SURFACE</div>
                <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "#fbbf24" }}>DAMP / DRYING</div>
            </div>
            <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "0.6rem", color: "#9ca3af", fontWeight: 800 }}>HUMIDITY</div>
                <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "#fff" }}>88%</div>
            </div>
            <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "0.6rem", color: "#9ca3af", fontWeight: 800 }}>RAIN PROB</div>
                <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "#dc2626" }}>15%</div>
            </div>
        </div>
    );

    // Strategic Outcome Banner
    const OutcomeBanner = ({ drivers }: { drivers: any[] }) => {
        if (drivers.length === 0) return null;

        const main = drivers[0];
        const impact = main.impact || 0;
        const improved = impact > 0;

        const formatTime = (seconds: number) => {
            if (!seconds) return "0:00.000";
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = (seconds % 60).toFixed(3);
            return h > 0 ? `${h}:${m}:${s.padStart(6, '0')}` : `${m}:${s.padStart(6, '0')}`;
        };

        return (
            <div style={{
                background: "rgba(16, 20, 28, 0.7)",
                backdropFilter: "blur(14px)",
                WebkitBackdropFilter: "blur(14px)",
                border: `1px solid ${improved ? "rgba(34, 197, 94, 0.5)" : "rgba(220, 38, 38, 0.5)"}`,
                boxShadow: `0 8px 32px 0 rgba(0, 0, 0, 0.8), inset 0 0 15px ${improved ? "rgba(34, 197, 94, 0.1)" : "rgba(220, 38, 38, 0.1)"}`,
                borderRadius: 12,
                padding: "16px 20px",
                marginBottom: 20,
                fontFamily: "var(--font-oxanium)",
                position: "relative",
                overflow: "hidden"
            }}>
                {/* Neon Accent Top Line */}
                <div style={{
                    position: "absolute",
                    top: 0, left: 0, right: 0, height: 2,
                    background: improved ? "linear-gradient(90deg, transparent, #22c55e, transparent)" : "linear-gradient(90deg, transparent, #dc2626, transparent)"
                }}></div>
                <div style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 10 }}>
                    <div style={{
                        fontSize: "1.8rem",
                        fontWeight: 900,
                        color: improved ? "#22c55e" : "#dc2626",
                        fontStyle: "italic",
                        letterSpacing: "1px"
                    }}>STRATEGIC ANALYSIS — TEAM FERRARI</div>

                    <div style={{ flex: 1, height: 2, background: "rgba(255,255,255,0.1)" }}></div>

                    <div style={{ textAlign: "right", display: "flex", alignItems: "center", gap: 10 }}>
                        <div style={{ marginRight: 10, textAlign: "right" }}>
                            <div style={{ fontSize: "0.7rem", color: "#9ca3af", fontWeight: 700 }}>AI STRATEGY</div>
                            <div style={{
                                fontSize: "1.5rem",
                                fontWeight: 900,
                                color: "#fff",
                                fontFamily: "var(--font-oxanium)",
                                background: main.call === "STAY OUT" ? "#22c55e" : "#E10600",
                                padding: "2px 12px",
                                borderRadius: 4,
                                display: "inline-block",
                                lineHeight: 1
                            }}>
                                {main.call}
                            </div>
                        </div>

                        <div style={{ textAlign: "right" }}>
                            <div style={{ fontSize: "0.7rem", color: "#9ca3af", fontWeight: 700 }}>AI CONFIDENCE</div>
                            <div style={{ fontSize: "2rem", fontWeight: 900, color: improved ? "#22c55e" : "#fff", fontFamily: "var(--font-oxanium)" }}>
                                {(main.prob * 100).toFixed(0)}%
                            </div>
                        </div>
                    </div>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1.5fr 1.5fr", gap: 20 }}>
                    {/* Status Column */}
                    <div>
                        <div style={{ fontSize: "0.75rem", color: "#9ca3af", fontWeight: 800, textTransform: "uppercase", marginBottom: 5 }}>Status</div>
                        <div style={{ fontSize: "1.1rem", fontWeight: 800, color: improved ? "#22c55e" : "#ffbdad", lineHeight: 1.2 }}>
                            {improved
                                ? `SUCCESS: AI recommendation gains +${impact.toFixed(3)}s vs History.`
                                : "WARNING: High Strategic Risk in Historical Decision."
                            }
                        </div>
                        <div style={{ fontSize: "0.85rem", marginTop: 6, color: "#ccc" }}>
                            {improved ? "Predicted outcome: Race Win (P1) achievable through optimized double-stack." : "Probability of losing position is high with historical data."}
                        </div>
                    </div>

                    {/* Driver Lap Comparison Column */}
                    <div style={{ background: "rgba(0,0,0,0.3)", padding: "12px 16px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.05)" }}>
                        <div style={{ fontSize: "0.7rem", color: "#9ca3af", fontWeight: 800, marginBottom: 12, display: "flex", justifyContent: "space-between" }}>
                            <span>LAP TIME COMPARISON</span>
                            <span>HISTORICAL vs WHAT-IF</span>
                        </div>
                        {drivers.slice(0, 2).map((d, i) => (
                            <div key={i} style={{ marginBottom: i === 0 ? 10 : 0, paddingBottom: i === 0 ? 10 : 0, borderBottom: i === 0 ? "1px solid rgba(255,255,255,0.1)" : "none" }}>
                                <div style={{ fontSize: "0.75rem", fontWeight: 800, color: i === 0 ? "#dc2626" : "#fbbf24", marginBottom: 4 }}>{d.driver === "LEC" ? "LECLERC" : "SAINZ"}</div>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                                    <span style={{ fontSize: "0.9rem", fontWeight: 700, fontFamily: "var(--font-oxanium)", color: "#888" }}>{formatTime(d.historical_lap_time)}</span>
                                    <span style={{ fontSize: "0.6rem", color: "#555" }}>➜</span>
                                    <span style={{ fontSize: "1rem", fontWeight: 900, color: "#22c55e", fontFamily: "var(--font-oxanium)" }}>{formatTime(d.predicted_lap_time)}</span>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Driver Race Comparison Column */}
                    <div style={{ background: "rgba(0,0,0,0.3)", padding: "12px 16px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.05)" }}>
                        <div style={{ fontSize: "0.7rem", color: "#9ca3af", fontWeight: 800, marginBottom: 12, display: "flex", justifyContent: "space-between" }}>
                            <span>TOTAL RACE TIME PROJECTION</span>
                            <span>HISTORICAL vs PREDICTED</span>
                        </div>
                        {drivers.slice(0, 2).map((d, i) => (
                            <div key={i} style={{ marginBottom: i === 0 ? 10 : 0, paddingBottom: i === 0 ? 10 : 0, borderBottom: i === 0 ? "1px solid rgba(255,255,255,0.1)" : "none" }}>
                                <div style={{ fontSize: "0.75rem", fontWeight: 800, color: i === 0 ? "#dc2626" : "#fbbf24", marginBottom: 4 }}>{d.driver === "LEC" ? "LECLERC" : "SAINZ"}</div>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                                    <span style={{ fontSize: "0.9rem", fontWeight: 700, fontFamily: "var(--font-oxanium)", color: "#888" }}>{formatTime(d.historical_race_time)}</span>
                                    <span style={{ fontSize: "0.6rem", color: "#555" }}>➜</span>
                                    <span style={{ fontSize: "1rem", fontWeight: 900, color: "#22c55e", fontFamily: "var(--font-oxanium)" }}>{formatTime(d.predicted_race_time)}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        );
    };

    // F1 Style Driver Card
    const DriverCard = ({ item, posNum, align = "left" }: { item: any; posNum: number; align?: "left" | "right" }) => {
        const driver = item.driver;
        const isLec = driver === "LEC";
        const primaryColor = isLec ? "#dc2626" : "#fbbf24";

        const firstName = isLec ? "CHARLES" : "CARLOS";
        const lastName = isLec ? "LECLERC" : "SAINZ";

        // Use historical data from API for the main card view
        const histLapSeconds = item.historical_lap_time || 84.5;
        const mins = Math.floor(histLapSeconds / 60);
        const secs = (histLapSeconds % 60).toFixed(3);
        const lapTime = `${mins}:${secs.padStart(6, '0')}`;

        const gap = item.telemetry?.gap_to_leader || 0;
        const finalGap = (gap === 0 || posNum === 1) ? "LEADER" : `+${gap.toFixed(3)}s`;

        // Prediction outcome for the card
        const predPos = item.telemetry?.predicted_position || posNum;

        const textAlign = align === "left" ? "left" : "right";
        const rowRev = align === "right" ? "row-reverse" : "row";

        // Tire Data
        const compound = item.telemetry?.compound || "INTERMEDIATE";
        const tireAge = item.telemetry?.tire_age || 0;

        return (
            <div style={{
                color: "#fff",
                fontFamily: "var(--font-oxanium)",
                padding: "20px",
                background: "rgba(16, 20, 28, 0.65)",
                backdropFilter: "blur(12px)",
                WebkitBackdropFilter: "blur(12px)",
                borderRadius: 16,
                border: `1px solid ${primaryColor}40`,
                boxShadow: `0 10px 40px rgba(0,0,0,0.6), inset 0 0 20px ${primaryColor}10`,
                position: "relative"
            }}>
                {/* Corner Neon Accent */}
                <div style={{
                    position: "absolute",
                    top: -1, [align]: -1, width: 40, height: 40,
                    borderTop: `2px solid ${primaryColor}`,
                    [align === "left" ? "borderLeft" : "borderRight"]: `2px solid ${primaryColor}`,
                    borderTopLeftRadius: align === "left" ? 16 : 0,
                    borderTopRightRadius: align === "right" ? 16 : 0,
                    boxShadow: `0 0 15px ${primaryColor}60`
                }}></div>
                {/* Header Row: Pos | Name | Logo */}
                <div style={{ display: "flex", flexDirection: rowRev, alignItems: "center", gap: 20, marginBottom: 20, borderBottom: "1px solid #333", paddingBottom: 10 }}>
                    {/* Position */}
                    <div style={{
                        fontSize: "4rem",
                        fontWeight: 900,
                        lineHeight: 1,
                        minWidth: 60,
                        textAlign: "center",
                        fontFamily: "var(--font-oxanium)"
                    }}>{posNum}</div>

                    {/* Name */}
                    <div style={{ flex: 1, textAlign: textAlign }}>
                        <div style={{ fontSize: "1rem", color: "#9ca3af", letterSpacing: "1px", fontWeight: 600 }}>DRIVER</div>
                        <div style={{ fontSize: "1.4rem", fontWeight: 500, fontStyle: "italic" }}>{firstName}</div>
                        <div style={{ fontSize: "2.4rem", fontWeight: 900, color: primaryColor, fontStyle: "italic", lineHeight: 0.9 }}>{lastName}</div>
                        <div style={{ fontSize: "1rem", color: "#ccc", marginTop: 4, fontWeight: 600 }}>FERRARI</div>
                    </div>

                    {/* Logo (Optional Image) */}
                    <div style={{
                        width: 50, height: 60,
                        background: "#fbbf24",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        clipPath: "polygon(0 0, 100% 0, 100% 85%, 50% 100%, 0 85%)",
                        color: "#000", fontSize: "1.5rem",
                        overflow: "hidden"
                    }}>
                        {item.logo_url ? (
                            <img
                                src={item.logo_url}
                                alt="Team Logo"
                                style={{
                                    width: "90%",
                                    height: "90%",
                                    objectFit: "contain",
                                    mixBlendMode: "multiply"
                                }}
                            />
                        ) : "F1"}
                    </div>
                </div>

                {/* Stats Row */}
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 15, flexDirection: rowRev, alignItems: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, flexDirection: rowRev }}>
                        <CompoundIcon compound={compound} />
                        <div style={{ textAlign: textAlign }}>
                            <div style={{ fontSize: "0.6rem", color: "#9ca3af", fontWeight: 800 }}>AGE</div>
                            <div style={{ fontSize: "1.2rem", fontWeight: 900, fontFamily: "var(--font-oxanium)" }}>{tireAge}L</div>
                        </div>
                    </div>

                    <div style={{ textAlign: textAlign }}>
                        <div style={{ fontSize: "0.8rem", color: "#9ca3af", fontWeight: 600 }}>LAP TIME (HIST)</div>
                        <div style={{ fontSize: "1.8rem", fontWeight: 900, fontStyle: "italic", fontFamily: "var(--font-oxanium)" }}>{lapTime}</div>
                    </div>
                </div>

                {/* Gap Row */}
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 15, flexDirection: rowRev, alignItems: "baseline" }}>
                    <div style={{ textAlign: textAlign }}>
                        <div style={{ fontSize: "0.8rem", color: "#9ca3af", fontWeight: 600 }}>GAP (HIST)</div>
                        <div style={{ fontSize: "1.8rem", fontWeight: 900, fontStyle: "italic", fontFamily: "var(--font-oxanium)" }}>{finalGap}</div>
                    </div>
                    <div style={{ textAlign: align === "left" ? "right" : "left", background: "rgba(34, 197, 94, 0.15)", padding: "4px 10px", borderRadius: 4, border: "1px solid #22c55e" }}>
                        <div style={{ fontSize: "0.6rem", color: "#22c55e", fontWeight: 800 }}>PREDICTED FINISH</div>
                        <div style={{ fontSize: "1.3rem", fontWeight: 900, color: "#fff", fontFamily: "var(--font-oxanium)" }}>P{predPos}</div>
                    </div>
                </div>

                {/* Bars */}
                <PerfBar label="FULL THROTTLE" value={item.telemetry?.throttle_pct || 81} color={primaryColor} align={align} />
                <PerfBar label="HEAVY BRAKING" value={item.telemetry?.brake_pct || 5} color={primaryColor} align={align} />
                <PerfBar label="CORNERING" value={item.telemetry?.cornering_pct || 14} color={primaryColor} align={align} />
            </div>
        );
    };

    // Strategy Transition Chart (Wet to Dry Crossover)
    const StrategyTransitionChart = () => {
        return (
            <div style={{
                background: "rgba(16, 20, 28, 0.45)",
                backdropFilter: "blur(12px)",
                borderRadius: 16,
                padding: "24px",
                border: "1px solid rgba(168, 85, 247, 0.2)",
                fontFamily: "var(--font-oxanium)",
                display: "grid",
                gridTemplateColumns: "1fr 1.5fr",
                gap: 40,
                alignItems: "center"
            }}>
                <div>
                    <div style={{ fontSize: "0.8rem", color: "#a855f7", fontWeight: 800, marginBottom: 8, letterSpacing: "2px", textTransform: "uppercase" }}>Strategic Analysis</div>
                    <div style={{ fontSize: "1.8rem", fontWeight: 900, color: "#fff", marginBottom: 12, lineHeight: 1.1 }}>THE <span style={{ color: "#a855f7" }}>MOMENT</span></div>
                    <div style={{ fontSize: "0.85rem", color: "#a6adbb", lineHeight: 1.5, borderLeft: "2px solid #a855f7", paddingLeft: 15 }}>
                        Analyzing defining strategic moments. Featuring the <b>2022 Monaco Crossover</b> (Ferrari Double Stack)
                        and 2024 highlights including Norris (Miami) and Hamilton (Silverstone).
                    </div>
                </div>

                <div style={{ position: "relative" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 15 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ width: 12, height: 2, background: "#22c55e", borderRadius: 1 }}></span>
                            <span style={{ fontSize: "0.6rem", fontWeight: 800, color: "#22c55e" }}>INTERS PACE</span>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ width: 12, height: 2, background: "#fbbf24", borderRadius: 1 }}></span>
                            <span style={{ fontSize: "0.6rem", fontWeight: 800, color: "#fbbf24" }}>SLICKS PACE</span>
                        </div>
                    </div>

                    <div style={{ height: 120, position: "relative", marginBottom: 15 }}>
                        <svg width="100%" height="100%" viewBox="0 0 300 100" preserveAspectRatio="none">
                            {/* Intermediate Pace (Decaying) */}
                            <path d="M 0 15 Q 150 25 300 95" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeDasharray="5 3" opacity="0.5" />
                            {/* Slick Pace (Rising) */}
                            <path d="M 0 95 Q 150 45 300 5" fill="none" stroke="#fbbf24" strokeWidth="4" />

                            {/* Crossover Window Glow */}
                            <rect x="130" y="0" width="40" height="100" fill="url(#windowGlow)" opacity="0.2" />

                            {/* Window Indicator */}
                            <line x1="150" y1="0" x2="150" y2="100" stroke="#a855f7" strokeWidth="2" strokeDasharray="4 4" />
                            <circle cx="150" cy="42" r="6" fill="#a855f7" filter="url(#mapGlow)" />

                            <defs>
                                <linearGradient id="windowGlow" x1="0" x2="0" y1="0" y2="1">
                                    <stop offset="0%" stopColor="#a855f7" />
                                    <stop offset="100%" stopColor="transparent" />
                                </linearGradient>
                            </defs>
                        </svg>
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", color: "#666", fontSize: "0.6rem", fontWeight: 800 }}>
                        <span>LAP 15</span>
                        <span style={{ color: "#a855f7", fontSize: "0.7rem" }}>LAP 21 (CROSSOVER)</span>
                        <span>LAP 28</span>
                    </div>
                </div>
            </div>
        );
    };



    const PerfBar = ({ label, value, color, align }: any) => {
        const roundedValue = Math.round(value);
        return (
            <div style={{ marginBottom: 12, fontFamily: "var(--font-oxanium)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: "#9ca3af", marginBottom: 4 }}>
                    <span style={{ fontWeight: 600, letterSpacing: "0.5px" }}>{align === "left" ? label : `${roundedValue}%`}</span>
                    <span style={{ fontWeight: 600, letterSpacing: "0.5px" }}>{align === "left" ? `${roundedValue}%` : label}</span>
                </div>
                <div style={{ width: "100%", height: 6, background: "#333", transform: align === "right" ? "scaleX(-1)" : "none", borderRadius: 3 }}>
                    <div style={{ width: `${roundedValue}%`, height: "100%", background: color, borderRadius: 3 }}></div>
                </div>
            </div>
        );
    };

    // Speed Graph F1 Style
    const SpeedDeltaAnalysis = ({ drivers, d1Data, d2Data, hoverX, setHoverX }: { drivers: any[], d1Data: any[], d2Data: any[], hoverX: number | null, setHoverX: (x: number | null) => void }) => {
        if (drivers.length < 2) return null;

        const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
            const svg = e.currentTarget;
            const rect = svg.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 1200;
            setHoverX(Math.max(0, Math.min(1200, x)));
        };

        const activePoint = hoverX !== null ? Math.floor(hoverX) : null;
        const lecSpeed = activePoint !== null ? d1Data[activePoint]?.speed : 0;
        const saiSpeed = activePoint !== null ? d2Data[activePoint]?.speed : 0;
        const delta = activePoint !== null ? (lecSpeed - saiSpeed) : 0;

        return (
            <div style={{
                marginTop: 30,
                fontFamily: "var(--font-oxanium)",
                background: "rgba(16, 20, 28, 0.7)",
                backdropFilter: "blur(12px)",
                WebkitBackdropFilter: "blur(12px)",
                padding: "24px",
                borderRadius: 16,
                border: "1px solid rgba(255,255,255,0.08)",
                boxShadow: "0 10px 50px rgba(0,0,0,0.8)",
                position: "relative"
            }}>
                <svg
                    width="100%" height="380" viewBox="0 -80 1200 420"
                    preserveAspectRatio="none"
                    onMouseMove={handleMouseMove}
                    onMouseLeave={() => setHoverX(null)}
                    style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, border: "1px solid rgba(255,255,255,0.05)", overflow: "visible", cursor: "crosshair" }}
                >
                    <defs>
                        <filter id="glowRed" x="-20%" y="-20%" width="140%" height="140%">
                            <feGaussianBlur stdDeviation="3" result="blur" />
                            <feComposite in="SourceGraphic" in2="blur" operator="over" />
                        </filter>
                        <filter id="glowYellow" x="-20%" y="-20%" width="140%" height="140%">
                            <feGaussianBlur stdDeviation="3" result="blur" />
                            <feComposite in="SourceGraphic" in2="blur" operator="over" />
                        </filter>
                    </defs>

                    {[50, 100, 150, 200, 250].map(y => (
                        <line key={y} x1="0" y1={y + 40} x2="1200" y2={y + 40} stroke="#1a1a1e" strokeWidth="1.5" />
                    ))}

                    {/* 1:1 Edge-to-Edge Visual Blocks */}
                    {[
                        { x1: 0, x2: 120 }, // T1-2
                        { x1: 220, x2: 275 }, // T3
                        { x1: 380, x2: 485 }, // T4-5
                        { x1: 600, x2: 705 }, // T6-7
                        { x1: 840, x2: 950 }, // T8-9
                        { x1: 1050, x2: 1105 }, // T10
                        { x1: 1180, x2: 1200 }, // T11
                    ].map((b, i) => (
                        <rect key={i} x={b.x1} y="0" width={b.x2 - b.x1} height="200" fill="#fff" opacity="0.05" />
                    ))}

                    {/* Broadcast Sector Slabs */}
                    {[
                        { label: "LOW SPEED", x: 120, w: 90 },
                        { label: "LOW SPEED", x: 450, w: 90 },
                        { label: "HIGH SPEED", x: 620, w: 105 },
                        { label: "MEDIUM SPEED", x: 860, w: 190 },
                        { label: "HIGH SPEED", x: 1120, w: 45 }
                    ].map((s, i) => (
                        <g key={i} transform={`translate(${s.x}, -60)`}>
                            <text x={s.w / 2} y="-12" fill="#9ca3af" fontSize="10" fontWeight="900" textAnchor="middle" fontFamily="var(--font-oxanium)" style={{ letterSpacing: "2.5px" }}>{s.label}</text>
                            <path d={`M 2 1 L 0 1 L 0 5 L ${s.w} 5 L ${s.w} 1 L ${s.w - 2} 1`} fill="none" stroke="#555" strokeWidth="1.5" />
                        </g>
                    ))}

                    {/* Grouped Turn Labels - 1:1 Calibration */}
                    {[
                        { label: "TURN 1  2", x: 60 },
                        { label: "3", x: 247 },
                        { label: "4  5", x: 432 },
                        { label: "6  7", x: 652 },
                        { label: "8  9", x: 895 },
                        { label: "10", x: 1077 },
                        { label: "11", x: 1190 }
                    ].map((l, i) => (
                        <text key={i} x={l.x} y={-25} fill="#fff" fontSize="13" fontWeight="900" textAnchor="middle" fontFamily="var(--font-oxanium)">
                            {l.label}
                        </text>
                    ))}

                    {/* 1:1 Duel Traces (Solid Red / Solid Yellow) - SYNCED MATH */}
                    <path
                        d={`M${d2Data.map(p => `${p.x},${200 - (p.speed / 320) * 160}`).join(" L")}`}
                        fill="none" stroke="#facc15" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" opacity="0.9"
                    />
                    <path
                        d={`M${d1Data.map(p => `${p.x},${200 - (p.speed / 320) * 160}`).join(" L")}`}
                        fill="none" stroke="#dc2626" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" filter="url(#glowRed)"
                    />

                    <text x="35" y="40" fill="#facc15" fontSize="13" fontWeight="900" transform="rotate(-90, 35, 40)" textAnchor="end" fontFamily="var(--font-oxanium)">304 KM/H</text>
                    <text x="35" y="195" fill="#facc15" fontSize="13" fontWeight="900" transform="rotate(-90, 35, 195)" textAnchor="end" fontFamily="var(--font-oxanium)">54 KM/H</text>
                    <text x="15" y="120" fill="#888" fontSize="10" fontWeight="900" transform="rotate(-90, 15, 120)" textAnchor="middle" fontFamily="var(--font-oxanium)" style={{ letterSpacing: "3px" }}>SPEED</text>

                    <g transform="translate(0, 230)">
                        {[
                            { x: 60, v: -0.103 },
                            { x: 247, v: -0.004 },
                            { x: 432, v: -0.021 },
                            { x: 652, v: -0.011 },
                            { x: 895, v: -0.111 },
                            { x: 1077, v: -0.156 },
                            { x: 1190, v: -0.321 }
                        ].map((d, i) => (
                            <text key={i} x={d.x} y={0} fill="#dc2626" fontSize="13" fontWeight="900" fontFamily="var(--font-oxanium)" textAnchor="middle">
                                {d.v.toFixed(3)}
                            </text>
                        ))}
                    </g>

                    <g transform="translate(0, 275)">
                        <rect x="0" y="0" width="1200" height="100" fill="#000" opacity="0.4" />
                        <line x1="0" y1="50" x2="1200" y2="50" stroke="#facc15" strokeWidth="1.5" opacity="0.6" />

                        {(() => {
                            let sum = 0;
                            const pathData = d1Data.map((p, idx) => {
                                sum += (p.speed - d2Data[idx].speed) * 0.012;
                                return `${p.x},${50 - sum}`;
                            }).join(" L");
                            return <path d={`M0,50 L${pathData}`} fill="none" stroke="#dc2626" strokeWidth="2.1" strokeLinecap="round" filter="url(#glowRed)" />;
                        })()}

                        <text x="50" y="20" fill="#fff" fontSize="15" fontWeight="900" opacity="0.1" fontFamily="var(--font-oxanium)" style={{ letterSpacing: "5px" }}>FASTER</text>
                        <text x="50" y="80" fill="#fff" fontSize="15" fontWeight="900" opacity="0.1" fontFamily="var(--font-oxanium)" style={{ letterSpacing: "5px" }}>SLOWER</text>
                        <text x="18" y="50" fill="#fff" fontSize="10" fontWeight="900" transform="rotate(-90, 18, 50)" textAnchor="middle" opacity="0.6" fontFamily="var(--font-oxanium)">DELTA</text>
                    </g>

                    {/* Interactive Hover Logic */}
                    {hoverX !== null && (
                        <g pointerEvents="none">
                            <line x1={hoverX} y1="-50" x2={hoverX} y2="400" stroke="#fff" strokeWidth="1" strokeDasharray="4 4" opacity="0.5" />
                            <circle cx={hoverX} cy={200 - (saiSpeed / 320) * 160} r="5" fill="#fbbf24" stroke="#fff" strokeWidth="2" />
                            <circle cx={hoverX} cy={200 - (lecSpeed / 320) * 160} r="5" fill="#dc2626" stroke="#fff" strokeWidth="2" />

                            <g transform={`translate(${hoverX > 1000 ? hoverX - 180 : hoverX + 20}, 50)`}>
                                <rect width="160" height="90" rx="8" fill="rgba(0,0,0,0.9)" stroke="rgba(255,255,255,0.2)" />
                                <text x="10" y="25" fill="#9ca3af" fontSize="10" fontWeight="800">LIVE TELEMETRY</text>
                                <text x="10" y="50" fill="#fff" fontSize="14" fontWeight="900">LEC: <tspan fill="#dc2626">{lecSpeed.toFixed(0)}</tspan> KM/H</text>
                                <text x="10" y="72" fill="#fff" fontSize="14" fontWeight="900">SAI: <tspan fill="#fbbf24">{saiSpeed.toFixed(0)}</tspan> KM/H</text>
                                <text x="110" y="25" fill={delta > 0 ? "#22c55e" : "#dc2626"} fontSize="12" fontWeight="900" textAnchor="end">
                                    {delta > 0 ? "+" : ""}{delta.toFixed(1)}
                                </text>
                            </g>
                        </g>
                    )}
                </svg>
            </div>
        );
    };

    return (
        <div className="section" style={{ background: "#101014", color: "#fff", padding: "20px", fontFamily: "var(--font-oxanium)" }}>
            <div style={{ display: "flex", alignItems: "center", borderBottom: "1px solid #333", paddingBottom: 15, marginBottom: 20 }}>
                <div style={{ fontSize: "2rem", fontWeight: 900, fontStyle: "italic", paddingRight: 20, borderRight: "1px solid #333", marginRight: 20 }}>F1</div>
                <div style={{ fontSize: "1.5rem", fontWeight: 700, fontStyle: "italic", textTransform: "uppercase" }}>MONACO — ANALYSIS</div>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 20 }}>
                <OutcomeBanner drivers={data} />
                <div style={{ marginBottom: 20 }}>
                    <TrackConditions />
                </div>
            </div>

            <div style={{ padding: "0 40px", marginBottom: 40 }}>
                {/* View Mode Toggle */}
                <div style={{
                    display: "flex",
                    justifyContent: "center",
                    marginBottom: 30,
                    gap: 10
                }}>
                    <button
                        onClick={() => setViewMode("2D")}
                        style={{
                            padding: "12px 32px",
                            fontSize: "1rem",
                            fontWeight: 900,
                            fontFamily: "var(--font-oxanium)",
                            background: viewMode === "2D" ? "linear-gradient(135deg, #17c3ff, #0ea5e9)" : "rgba(23, 195, 255, 0.1)",
                            color: viewMode === "2D" ? "#000" : "#17c3ff",
                            border: `2px solid ${viewMode === "2D" ? "#17c3ff" : "rgba(23, 195, 255, 0.3)"}`,
                            borderRadius: 8,
                            cursor: "pointer",
                            transition: "all 0.3s ease",
                            textTransform: "uppercase",
                            letterSpacing: "1px",
                            boxShadow: viewMode === "2D" ? "0 0 20px rgba(23, 195, 255, 0.5)" : "none"
                        }}
                    >
                        2D MAP
                    </button>
                    <button
                        onClick={() => setViewMode("3D")}
                        style={{
                            padding: "12px 32px",
                            fontSize: "1rem",
                            fontWeight: 900,
                            fontFamily: "var(--font-oxanium)",
                            background: viewMode === "3D" ? "linear-gradient(135deg, #a855f7, #9333ea)" : "rgba(168, 85, 247, 0.1)",
                            color: viewMode === "3D" ? "#fff" : "#a855f7",
                            border: `2px solid ${viewMode === "3D" ? "#a855f7" : "rgba(168, 85, 247, 0.3)"}`,
                            borderRadius: 8,
                            cursor: "pointer",
                            transition: "all 0.3s ease",
                            textTransform: "uppercase",
                            letterSpacing: "1px",
                            boxShadow: viewMode === "3D" ? "0 0 20px rgba(168, 85, 247, 0.5)" : "none"
                        }}
                    >
                        3D VIEW
                    </button>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1.8fr 1fr", gap: 50, alignItems: "start" }}>
                    {data.length > 0 && (
                        <div style={{ transform: "scale(1.1)", transformOrigin: "top left" }}>
                            <DriverCard item={data[0]} posNum={data[0].telemetry?.position || 4} align="left" />
                        </div>
                    )}

                    {/* Conditional rendering based on viewMode */}
                    {viewMode === "2D" ? (
                        <MonacoMap drivers={data} d1Data={d1Data} d2Data={d2Data} hoverX={hoverX} />
                    ) : (
                        <Monaco3DTrack d1Data={d1Data} d2Data={d2Data} hoverX={hoverX} />
                    )}

                    {data.length > 1 && (
                        <div style={{ transform: "scale(1.1)", transformOrigin: "top right" }}>
                            <DriverCard item={data[1]} posNum={data[1].telemetry?.position || 2} align="right" />
                        </div>
                    )}
                </div>
            </div>

            {/* Strategic Analysis Layer */}
            <div style={{ padding: "0 40px", marginBottom: 50 }}>
                <StrategyTransitionChart />
            </div>

            <SpeedDeltaAnalysis
                drivers={data}
                d1Data={d1Data}
                d2Data={d2Data}
                hoverX={hoverX}
                setHoverX={setHoverX}
            />
        </div>
    );
}
