"use client";

import React from "react";

interface DriverComparisonProps {
    driver1: {
        code: string;
        team: string;
        teamColor: string;
        lapTime: string;
        s1: string;
        s2: string;
        s3: string;
        topSpeed: number;
        tireAge: number;
        tireCompound: string;
    };
    driver2: {
        code: string;
        team: string;
        teamColor: string;
        lapTime: string;
        s1: string;
        s2: string;
        s3: string;
        topSpeed: number;
        tireAge: number;
        tireCompound: string;
    };
}

export default function DriverComparisonCard({ driver1, driver2 }: DriverComparisonProps) {
    const DriverPanel = ({ driver, position }: { driver: any; position: "left" | "right" }) => (
        <div style={{
            flex: 1,
            padding: 20,
            background: `linear-gradient(135deg, ${position === "left" ? "rgba(225,6,0,0.1)" : "rgba(23,195,255,0.1)"}, rgba(15,21,31,0.9))`,
            borderRadius: 12,
            border: `2px solid ${driver.teamColor}`,
            position: "relative",
            overflow: "hidden"
        }}>
            {/* Team Color Bar */}
            <div style={{
                position: "absolute",
                top: 0,
                [position]: 0,
                width: 4,
                height: "100%",
                background: driver.teamColor,
                boxShadow: `0 0 15px ${driver.teamColor}`
            }}></div>

            {/* Driver Code */}
            <div style={{
                fontFamily: "var(--font-oxanium)",
                fontSize: "2.5rem",
                fontWeight: 900,
                letterSpacing: "0.05em",
                marginBottom: 4,
                color: driver.teamColor,
                textShadow: `0 0 20px ${driver.teamColor}`
            }}>
                {driver.code}
            </div>

            {/* Team Name */}
            <div style={{
                fontSize: "0.7rem",
                color: "#a6adbb",
                textTransform: "uppercase",
                letterSpacing: "0.15em",
                marginBottom: 20
            }}>
                {driver.team}
            </div>

            {/* Lap Time */}
            <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: "0.7rem", color: "#6e7888", marginBottom: 4 }}>BEST LAP</div>
                <div style={{
                    fontFamily: "var(--font-oxanium)",
                    fontSize: "1.8rem",
                    fontWeight: 800,
                    color: "#FFD700"
                }}>
                    {driver.lapTime}
                </div>
            </div>

            {/* Sector Times */}
            <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
                <div style={{ flex: 1 }}>
                    <div style={{ fontSize: "0.65rem", color: "#FF2B2B", marginBottom: 2 }}>S1</div>
                    <div style={{ fontSize: "0.9rem", fontWeight: 700 }}>{driver.s1}</div>
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{ fontSize: "0.65rem", color: "#8B5CF6", marginBottom: 2 }}>S2</div>
                    <div style={{ fontSize: "0.9rem", fontWeight: 700 }}>{driver.s2}</div>
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{ fontSize: "0.65rem", color: "#F1C232", marginBottom: 2 }}>S3</div>
                    <div style={{ fontSize: "0.9rem", fontWeight: 700 }}>{driver.s3}</div>
                </div>
            </div>

            {/* Stats */}
            <div style={{ display: "flex", justifyContent: "space-between", paddingTop: 12, borderTop: "1px solid rgba(255,255,255,0.1)" }}>
                <div>
                    <div style={{ fontSize: "0.65rem", color: "#6e7888" }}>TOP SPEED</div>
                    <div style={{ fontSize: "1rem", fontWeight: 700 }}>{driver.topSpeed} km/h</div>
                </div>
                <div>
                    <div style={{ fontSize: "0.65rem", color: "#6e7888" }}>TIRE AGE</div>
                    <div style={{ fontSize: "1rem", fontWeight: 700 }}>
                        {driver.tireAge} <span style={{ fontSize: "0.7rem", color: getTireColor(driver.tireCompound) }}>{driver.tireCompound}</span>
                    </div>
                </div>
            </div>
        </div>
    );

    const getTireColor = (compound: string) => {
        switch (compound.toUpperCase()) {
            case "S": return "#E10600";
            case "M": return "#FFD700";
            case "H": return "#FFFFFF";
            default: return "#17C3FF";
        }
    };

    return (
        <div className="card" style={{ padding: 24, marginTop: 24 }}>
            <div style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 16,
                marginBottom: 20
            }}>
                <div style={{
                    fontSize: "1.2rem",
                    fontWeight: 800,
                    textTransform: "uppercase",
                    letterSpacing: "0.1em",
                    background: "linear-gradient(90deg, #E10600, #17C3FF)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent"
                }}>
                    DRIVER COMPARISON
                </div>
            </div>

            <div style={{ display: "flex", gap: 16, position: "relative" }}>
                <DriverPanel driver={driver1} position="left" />

                {/* VS Divider */}
                <div style={{
                    position: "absolute",
                    left: "50%",
                    top: "50%",
                    transform: "translate(-50%, -50%)",
                    width: 60,
                    height: 60,
                    background: "linear-gradient(135deg, #E10600, #17C3FF)",
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontFamily: "var(--font-oxanium)",
                    fontSize: "1.4rem",
                    fontWeight: 900,
                    color: "#fff",
                    boxShadow: "0 0 30px rgba(225, 6, 0, 0.6)",
                    zIndex: 10,
                    border: "3px solid #000"
                }}>
                    VS
                </div>

                <DriverPanel driver={driver2} position="right" />
            </div>
        </div>
    );
}
