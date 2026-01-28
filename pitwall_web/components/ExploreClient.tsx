"use client";

import { useEffect, useState, useCallback } from "react";
import { motion } from 'framer-motion';
import { Play, RotateCcw, Zap, TrendingUp, AlertTriangle, ArrowRight } from 'lucide-react';
import ImpactChart from './ImpactChart';
import { getDemoContext, getDemoDecision, getDemoTelemetry, getDemoPitWindow, getDemoImpact } from "../lib/api";
import ContextChips from "./ContextChips";
import TelemetrySection from "./TelemetrySection";
import PitWindowSection from "./PitWindowSection";
import StrategySection from "./StrategySection";
import BroadcastHeader from "./BroadcastHeader";
import LiveTicker from "./LiveTicker";

interface Meta {
    drivers: string[];
    circuits: string[];
    weather_vals: string[];
    years: string[];
    lap_min?: number;
    lap_max?: number;
}

export default function ExploreClient() {
    // State
    const [loading, setLoading] = useState(false);
    const [meta, setMeta] = useState<Meta>({ drivers: [], circuits: [], weather_vals: [], years: [] });

    // Selection State
    const [dataset, setDataset] = useState<"my" | "ref">("my");
    const [year, setYear] = useState<string>("");
    const [driver, setDriver] = useState<string>("");
    const [circuit, setCircuit] = useState<string>("");
    const [weather, setWeather] = useState<string>("");
    const [lap, setLap] = useState<number>(1);

    // Data State
    const [decision, setDecision] = useState<any>(null);
    const [telemetry, setTelemetry] = useState<any>(null);
    const [windowInfo, setWindowInfo] = useState<any>(null);
    const [impact, setImpact] = useState<any>(null);
    const [error, setError] = useState<string>("");

    // Initial Load & context refresh
    const fetchContext = useCallback(async (isInit = false) => {
        try {
            const res = await getDemoContext(dataset, isInit ? undefined : driver, isInit ? undefined : circuit, isInit ? undefined : weather, isInit ? undefined : year);

            setMeta(prev => ({
                ...prev,
                drivers: res.drivers || [],
                circuits: res.circuits || [],
                weather_vals: res.weather_vals || prev.weather_vals,
                years: res.years || prev.years,
                lap_min: res.lap_min ?? prev.lap_min,
                lap_max: res.lap_max ?? prev.lap_max
            }));

            if (!isInit) {
                if (res.drivers && !res.drivers.includes(driver)) {
                    setDriver(res.drivers[0] || "");
                }
                if (res.weather_vals && !res.weather_vals.includes(weather)) {
                    setWeather(res.weather_vals[0] || "Dry");
                }
            }

            if (isInit) {
                const sel = res.selection || {};
                if (sel.year) setYear(sel.year);
                if (sel.driver) setDriver(sel.driver);
                if (sel.circuit) setCircuit(sel.circuit);
                if (sel.weather) setWeather(sel.weather);
                if (sel.lap) setLap(Number(sel.lap));
            }
        } catch (e: any) {
            setError(e.message);
        }
    }, [dataset, driver, circuit, weather, year]);

    useEffect(() => {
        fetchContext(true);
    }, []);

    useEffect(() => {
        if (year || circuit) {
            fetchContext(false);
        }
    }, [year, circuit]);

    // Run Scenario
    const runScenario = async () => {
        setLoading(true);
        setError("");
        try {
            const sel = { dataset, driver, circuit, weather, lap, year };
            const [decData, telData, winData, impData] = await Promise.all([
                getDemoDecision(sel),
                getDemoTelemetry(sel),
                getDemoPitWindow(sel),
                getDemoImpact(sel)
            ]);

            setDecision(decData);
            setTelemetry(telData);
            setWindowInfo(winData);
            setImpact(impData);
        } catch (err: any) {
            setError(err.message || "Failed to run scenario");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ maxWidth: 1400, margin: "0 auto", paddingBottom: 60, fontFamily: "var(--font-oxanium)" }}>
            <BroadcastHeader />
            <LiveTicker />

            <div className="section">
                <div className="section-title">Scenario Control</div>

                {error && (
                    <div style={{ background: "rgba(255,0,0,0.1)", border: "1px solid red", padding: 10, borderRadius: 8, marginBottom: 20, color: "red" }}>
                        {error}
                    </div>
                )}

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    style={{ marginTop: 40 }}
                >
                    <div
                        style={{
                            background: "rgba(16, 20, 28, 0.8)",
                            backdropFilter: "blur(12px)",
                            padding: "24px",
                            borderRadius: 4,
                            borderTop: "2px solid #E10600",
                            borderBottom: "1px solid rgba(255, 255, 255, 0.05)",
                            maxWidth: 1000,
                            margin: "0 auto"
                        }}
                    >
                        <div style={{ display: "flex", flexDirection: "column", gap: 30 }}>
                            {/* Row 1: Inputs */}
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 20 }}>
                                <div>
                                    <label style={{ display: "block", marginBottom: 8, fontSize: "0.75rem", color: "#888", fontWeight: 700, letterSpacing: "1px" }}>SEASON</label>
                                    <select value={year} onChange={(e) => setYear(e.target.value)} style={{ width: "100%", padding: "10px", background: "#0a0a0c", border: "1px solid #333", color: "#fff", borderRadius: 4, fontFamily: "var(--font-oxanium)", fontSize: "0.85rem" }}>
                                        {meta.years?.filter(y => y !== "2022").map(y => <option key={y} value={y}>{y}</option>)}
                                    </select>
                                </div>
                                <div>
                                    <label style={{ display: "block", marginBottom: 8, fontSize: "0.75rem", color: "#888", fontWeight: 700, letterSpacing: "1px" }}>DRIVER</label>
                                    <select value={driver} onChange={(e) => setDriver(e.target.value)} style={{ width: "100%", padding: "12px", background: "#0a0a0c", border: "1px solid #333", color: "#fff", borderRadius: 8, fontFamily: "var(--font-oxanium)", fontSize: "0.9rem" }}>
                                        {meta.drivers.map(d => <option key={d} value={d}>{d}</option>)}
                                    </select>
                                </div>
                                <div>
                                    <label style={{ display: "block", marginBottom: 8, fontSize: "0.75rem", color: "#888", fontWeight: 700, letterSpacing: "1px" }}>CIRCUIT</label>
                                    <select value={circuit} onChange={(e) => setCircuit(e.target.value)} style={{ width: "100%", padding: "12px", background: "#0a0a0c", border: "1px solid #333", color: "#fff", borderRadius: 8, fontFamily: "var(--font-oxanium)", fontSize: "0.9rem" }}>
                                        {meta.circuits.map(c => <option key={c} value={c}>{c}</option>)}
                                    </select>
                                </div>
                                <div>
                                    <label style={{ display: "block", marginBottom: 8, fontSize: "0.75rem", color: "#888", fontWeight: 700, letterSpacing: "1px" }}>WEATHER</label>
                                    <select value={weather} onChange={(e) => setWeather(e.target.value)} style={{ width: "100%", padding: "12px", background: "#0a0a0c", border: "1px solid #333", color: "#fff", borderRadius: 8, fontFamily: "var(--font-oxanium)", fontSize: "0.9rem" }}>
                                        {meta.weather_vals.map(w => <option key={w} value={w}>{w}</option>)}
                                    </select>
                                </div>
                            </div>

                            {/* Row 2: Split Controls & Context */}
                            <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: 30, alignItems: "start" }}>
                                {/* Left Column: Slider & Action */}
                                <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                                    <div>
                                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                                            <label style={{ fontSize: "0.7rem", color: "#888", fontWeight: 700, letterSpacing: "1px" }}>RACE PROGRESS</label>
                                            <span style={{ fontSize: "0.85rem", color: "#E10600", fontWeight: 900 }}>LAP {lap}</span>
                                        </div>
                                        <input type="range" min={meta.lap_min || 1} max={meta.lap_max || 70} value={lap} onChange={(e) => setLap(Number(e.target.value))} style={{ width: "100%", cursor: "pointer", accentColor: "#E10600" }} />
                                    </div>

                                    <button
                                        onClick={runScenario}
                                        disabled={loading}
                                        style={{
                                            width: "100%", padding: "14px",
                                            background: loading ? "#2a2a2a" : "#E10600",
                                            color: "#fff", border: "none", borderRadius: 4,
                                            fontWeight: 900, fontSize: "1rem",
                                            cursor: loading ? "not-allowed" : "pointer",
                                            textTransform: "uppercase", letterSpacing: "2px",
                                            fontFamily: "var(--font-oxanium)",
                                            display: "flex", justifyContent: "center", alignItems: "center", gap: 10,
                                            transition: "all 0.2s ease"
                                        }}
                                    >
                                        {loading ? <RotateCcw className="derive-spin" size={18} /> : <Play size={18} fill="currentColor" />}
                                        {loading ? "CALCULATE STRATEGY" : "RUN SIMULATION"}
                                    </button>
                                </div>

                                {/* Right Column: Context Box */}
                                <div style={{
                                    background: "#080a0f",
                                    border: "1px solid #1f2937",
                                    borderRadius: 6,
                                    padding: "20px",
                                    height: "100%",
                                    display: "flex",
                                    flexDirection: "column",
                                    justifyContent: "center"
                                }}>
                                    <div style={{ fontSize: "0.7rem", color: "#17C3FF", fontWeight: 800, letterSpacing: "1px", marginBottom: 12, textTransform: "uppercase" }}>
                                        Live Telemetry
                                    </div>
                                    <ContextChips ctx={{
                                        sc_active: telemetry?.test_row?.sc_active,
                                        vsc_active: telemetry?.test_row?.vsc_active,
                                        rain: weather === "Wet" || telemetry?.test_row?.rain > 0,
                                        track_temp: telemetry?.test_row?.track_temp,
                                        tire_age: telemetry?.test_row?.tire_age || telemetry?.test_row?.tyre_age,
                                        position: telemetry?.test_row?.position,
                                        gap: telemetry?.test_row?.gap_to_leader
                                    }} />
                                </div>
                            </div>
                        </div>
                    </div>
                </motion.div>

                {decision && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                        className="animate-fade-in"
                        style={{ marginTop: 40 }}
                    >
                        {/* ... (Header for Split) */}

                        <OutcomeBanner
                            trainDec={decision.model}
                            testDec={decision.historical}
                            impactSec={impact?.net_gain_sec || 0}
                            driver={driver}
                        />

                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 30, marginTop: 30, alignItems: "stretch" }}>
                            <DriverCard
                                label="AI RECOMMENDATION (80% TRAIN)"
                                color="#E10600"
                                driver={driver}
                                item={decision.model}
                                telemetry={telemetry?.train_row}
                                circuit={circuit}
                            />
                            <DriverCard
                                label="HISTORICAL TRUTH (GROUND TRUTH)"
                                color="#17C3FF"
                                driver={driver}
                                item={decision.historical}
                                telemetry={telemetry?.test_row}
                                circuit={circuit}
                            />
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1.2fr", gap: 30, marginTop: 30 }}>
                            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                                <StrategySection
                                    modelCall={decision.model?.payload?.decision}
                                    decisionSentence={`AI REASONING: ${decision.model?.sentence || "..."}`}
                                    impact={impact?.net_gain_sec || 0}
                                />
                            </div>

                            <ImpactChart
                                aiDecision={decision.model?.payload?.decision}
                                histDecision={decision.historical?.payload?.decision}
                                impact={impact?.net_gain_sec || 0}
                            />
                        </div>

                        <div style={{ marginTop: 30 }}>
                            <PitWindowSection
                                start={windowInfo?.window_start}
                                end={windowInfo?.window_end}
                                currentLap={lap}
                                text={windowInfo?.pit_window_text || "Predicted Window"}
                            />
                        </div>

                        <div style={{ marginTop: 30, background: "rgba(16, 20, 28, 0.5)", border: "1px solid rgba(255, 255, 255, 0.05)", borderRadius: 12, padding: "24px" }}>
                            <div style={{ display: "flex", gap: 30 }}>
                                <div style={{ flex: 1 }}>
                                    <div style={{ fontSize: "0.7rem", color: "#E10600", fontWeight: 800, marginBottom: 8, letterSpacing: "1px" }}>STRATEGIC LOGIC EXPLORER</div>
                                    <h3 style={{ fontSize: "1.2rem", color: "#fff", marginBottom: 12 }}>Dual-Layer AI Decision Engine</h3>
                                    <p style={{ fontSize: "0.85rem", color: "#a6adbb", lineHeight: 1.6, margin: 0 }}>
                                        The PitWall AI uses a two-stage verification process to ensure strategy safety:
                                    </p>
                                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 15 }}>
                                        <div>
                                            <div style={{ color: "#E10600", fontWeight: 900, fontSize: "0.75rem", marginBottom: 4 }}>LAYER 1: XGBOOST MODEL</div>
                                            <p style={{ fontSize: "0.8rem", color: "#888", margin: 0 }}>Learns from historical data (What did teams do in the past?). Provides the base probability.</p>
                                        </div>
                                        <div>
                                            <div style={{ color: "#22c55e", fontWeight: 900, fontSize: "0.75rem", marginBottom: 4 }}>LAYER 2: SAFETY POLICY</div>
                                            <p style={{ fontSize: "0.8rem", color: "#888", margin: 0 }}>Real-time optimization. Overrides the model if it detects significant Net Gain or critical tire wear.</p>
                                        </div>
                                    </div>
                                </div>
                                <div style={{ width: 200, background: "rgba(225, 6, 0, 0.05)", borderRadius: 8, padding: 15, border: "1px solid rgba(225, 6, 0, 0.2)" }}>
                                    <div style={{ fontSize: "0.6rem", color: "#888", marginBottom: 10 }}>AI STATE</div>
                                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem" }}>
                                            <span style={{ color: "#aaa" }}>Threshold:</span>
                                            <span style={{ color: "#fff", fontWeight: 700 }}>{(decision?.policy?.threshold * 100).toFixed(0)}%</span>
                                        </div>
                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem" }}>
                                            <span style={{ color: "#aaa" }}>Margin:</span>
                                            <span style={{ color: "#fff", fontWeight: 700 }}>±5%</span>
                                        </div>
                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem" }}>
                                            <span style={{ color: "#aaa" }}>Lookahead:</span>
                                            <span style={{ color: "#fff", fontWeight: 700 }}>10 Laps</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}
            </div>
        </div>
    );
}

// --- High Fidelity Internal Components ---

function OutcomeBanner({ trainDec, testDec, impactSec, driver }: any) {
    const improved = impactSec >= 0;
    const displayGain = Math.abs(impactSec);

    const SplitBadge = ({ label, decision, prob, color, source }: any) => {
        const isPolicy = source === "POLICY";
        return (
            <div style={{
                background: "rgba(0,0,0,0.3)",
                padding: "16px 20px",
                borderRadius: 12,
                border: `1px solid ${color}${isPolicy ? '80' : '40'}`,
                display: "flex",
                flexDirection: "column",
                gap: 12,
                flex: 1,
                position: "relative",
                overflow: "hidden"
            }}>
                {isPolicy && (
                    <div style={{
                        position: "absolute",
                        top: 0,
                        right: 0,
                        background: color,
                        color: "#fff",
                        fontSize: "0.55rem",
                        fontWeight: 900,
                        padding: "2px 8px",
                        borderBottomLeftRadius: 8,
                        textTransform: "uppercase",
                        letterSpacing: "1px",
                        boxShadow: `0 2px 10px ${color}40`
                    }}>
                        Policy Override
                    </div>
                )}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                    <span style={{ fontSize: "0.7rem", fontWeight: 800, color: "#9ca3af", letterSpacing: "1px" }}>{label}</span>
                    <span style={{ fontSize: "1.2rem", fontWeight: 900, color: color }}>{(prob * 100).toFixed(0)}%</span>
                </div>
                <div style={{ fontSize: "1.6rem", fontWeight: 900, color: "#fff", fontStyle: "italic" }}>
                    {decision || "STAY OUT"}
                </div>
                <div style={{ height: 4, width: "100%", background: "#111", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${prob * 100}%`, background: color }}></div>
                </div>
            </div>
        );
    };

    return (
        <div style={{
            background: "rgba(16, 20, 28, 0.75)",
            backdropFilter: "blur(16px)",
            WebkitBackdropFilter: "blur(16px)",
            border: "1px solid rgba(255, 255, 255, 0.08)",
            boxShadow: "0 20px 60px rgba(0, 0, 0, 0.6)",
            borderRadius: 16,
            padding: "24px 30px",
            marginBottom: 20,
            fontFamily: "var(--font-oxanium)",
            position: "relative"
        }}>
            <div style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 20 }}>
                <div style={{ fontSize: "1.5rem", fontWeight: 900, color: "#fff", fontStyle: "italic", letterSpacing: "1px" }}>
                    DUAL-SPLIT ANALYSIS: {driver}
                </div>
                <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.06)" }}></div>
                <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: "0.7rem", color: "#9ca3af", fontWeight: 800 }}>ESTIMATED IMPACT</div>
                    <div style={{ fontSize: "1.8rem", fontWeight: 900, color: improved ? "#22c55e" : "#ff5c5c" }}>
                        {improved ? "+" : "-"}{displayGain.toFixed(3)}s
                    </div>
                </div>
            </div>

            <div style={{ display: "flex", gap: 20 }}>
                <SplitBadge
                    label="AI RECOMMENDATION (80% TRAIN)"
                    decision={trainDec?.payload?.decision}
                    prob={trainDec?.proba || 0.5}
                    color="#E10600"
                    source={trainDec?.payload?.decision_source}
                />
                <SplitBadge
                    label="HISTORICAL TRUTH (REAL WORLD)"
                    decision={testDec?.payload?.decision}
                    prob={testDec?.proba || 1.0}
                    color="#17C3FF"
                    source="HISTORIC"
                />
            </div>

            <div style={{ marginTop: 16, fontSize: "0.85rem", color: "#fff", fontStyle: "italic", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 12, display: "flex", alignItems: "flex-start", gap: 10 }}>
                <Zap size={14} style={{ color: "#E10600", marginTop: 2, flexShrink: 0 }} />
                <span>
                    {trainDec?.sentence || "Comparison: AI Recommendation (Model Knowledge) vs. Historical Reality (Ground Truth)."}
                </span>
            </div>
        </div>
    );
}

function DriverCard({ label, color, driver, item, telemetry, circuit }: any) {
    const isBox = item?.payload?.decision !== "STAY OUT";
    const primaryColor = color || (isBox ? "#22c55e" : "#17C3FF");
    const compound = (telemetry?.compound && telemetry.compound.toUpperCase() !== "UNKNOWN") ? telemetry.compound : "MEDIUM";
    const tireAge = telemetry?.tire_age || telemetry?.stint_laps || 0;

    return (
        <div style={{
            color: "#fff",
            fontFamily: "var(--font-oxanium)",
            padding: "24px",
            background: "rgba(16, 20, 28, 0.7)",
            backdropFilter: "blur(12px)",
            borderRadius: 16,
            border: `1px solid ${primaryColor}40`,
            boxShadow: `0 10px 40px rgba(0,0,0,0.6), inset 0 0 20px ${primaryColor}10`,
            position: "relative"
        }}>
            {label && (
                <div style={{
                    position: "absolute",
                    top: -12,
                    left: 20,
                    background: color,
                    padding: "2px 10px",
                    borderRadius: 4,
                    fontSize: "0.6rem",
                    fontWeight: 900,
                    letterSpacing: "1px"
                }}>
                    {label}
                </div>
            )}
            <div style={{ position: "absolute", top: -1, left: -1, width: 40, height: 40, borderTop: `2px solid ${primaryColor}`, borderLeft: `2px solid ${primaryColor}`, borderTopLeftRadius: 16, boxShadow: `0 0 15px ${primaryColor}60` }}></div>
            <div style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 24, borderBottom: "1px solid #333", paddingBottom: 16 }}>
                <div style={{ fontSize: "3.5rem", fontWeight: 900, lineHeight: 1, minWidth: 60 }}>{telemetry?.position || "-"}</div>
                <div style={{ flex: 1 }}>
                    <div style={{ fontSize: "0.9rem", color: "#9ca3af", fontWeight: 600 }}>CURRENT DRIVER</div>
                    <div style={{ fontSize: "2rem", fontWeight: 900, color: primaryColor, fontStyle: "italic", lineHeight: 1 }}>{driver}</div>
                    <div style={{ fontSize: "0.9rem", color: "#ccc", marginTop: 4, fontWeight: 600 }}>{circuit}</div>
                </div>
                <div style={{ width: 50, height: 50, background: "#333", display: "flex", alignItems: "center", justifyContent: "center", borderRadius: "50%", border: `2px solid ${primaryColor}`, fontWeight: 900 }}>F1</div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
                <div>
                    <div style={{ fontSize: "0.7rem", color: "#9ca3af", fontWeight: 800 }}>TYRE COMPOUND</div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
                        <div style={{ width: 24, height: 24, borderRadius: "50%", border: "2px solid #ccc", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.7rem", fontWeight: 900 }}>{compound[0]}</div>
                        <div style={{ fontSize: "1.1rem", fontWeight: 900 }}>{compound}</div>
                    </div>
                </div>
                <div>
                    <div style={{ fontSize: "0.7rem", color: "#9ca3af", fontWeight: 800 }}>TYRE AGE / WEAR</div>
                    <div style={{ fontSize: "1.1rem", fontWeight: 900, marginTop: 4 }}>{tireAge.toFixed(0)} Laps</div>
                    <div style={{ height: 4, width: "100%", background: "#333", borderRadius: 2, marginTop: 6, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${telemetry?.tire_wear_pct ? (Number(telemetry.tire_wear_pct) * 100) : (tireAge / 35 * 100)}%`, background: primaryColor }}></div>
                    </div>
                </div>
            </div>

            <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: "0.7rem", color: "#9ca3af", fontWeight: 800, marginBottom: 8 }}>PERFORMANCE METRICS</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    <div style={{ height: 6, background: "#333", borderRadius: 3 }}>
                        <div style={{ width: `${telemetry?.throttle || 0}%`, height: "100%", background: "#E10600", borderRadius: 3, transition: "width 0.3s ease" }}></div>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.6rem", color: "#888" }}>
                        <span>THROTTLE</span>
                        <span>{(telemetry?.throttle || 0).toFixed(1)}%</span>
                    </div>
                </div>
            </div>

            <div style={{ background: "rgba(34, 197, 94, 0.1)", padding: "12px", borderRadius: 8, border: "1px solid rgba(34, 197, 94, 0.3)" }}>
                <div style={{ fontSize: "0.6rem", color: "#22c55e", fontWeight: 800 }}>AI POSITION FORECAST</div>
                <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "#fff" }}>P{telemetry?.position || "-"} (STABLE)</div>
            </div>
        </div>
    );
}

function StrategyTransitionChart({ isPitRecommended, netGain }: any) {
    return (
        <div style={{ background: "rgba(16, 20, 28, 0.45)", backdropFilter: "blur(12px)", borderRadius: 16, padding: "24px", border: "1px solid rgba(168, 85, 247, 0.2)", fontFamily: "var(--font-oxanium)", display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: 30, alignItems: "center", height: "100%" }}>
            <div>
                <div style={{ fontSize: "0.7rem", color: "#a855f7", fontWeight: 800, marginBottom: 6, letterSpacing: "2px", textTransform: "uppercase" }}>Strategic Pulse</div>
                <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "#fff", marginBottom: 8, lineHeight: 1.1 }}>THE <span style={{ color: "#a855f7" }}>DECISION</span> WINDOW</div>
                <p style={{ fontSize: "0.8rem", color: "#a6adbb", lineHeight: 1.4, margin: 0 }}>
                    {isPitRecommended ? "Pit stop opportunity detected. Strategic gain outweighs the loss." : "Staying out is optimal. Current track position and tire health provide better race time."}
                </p>
            </div>
            <div style={{ position: "relative" }}>
                <div style={{ height: 100, position: "relative" }}>
                    <svg width="100%" height="100%" viewBox="0 0 300 100" preserveAspectRatio="none">
                        <path d="M 0 80 Q 150 50 300 20" fill="none" stroke="#E10600" strokeWidth="2" strokeDasharray="4 4" opacity="0.4" />
                        <path d={`M 0 50 Q 150 ${isPitRecommended ? 20 : 80} 300 ${isPitRecommended ? 10 : 90}`} fill="none" stroke="#22c55e" strokeWidth="3" />
                        <line x1="150" y1="0" x2="150" y2="100" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                        <circle cx="150" cy={isPitRecommended ? 20 : 80} r="4" fill="#a855f7" />
                    </svg>
                    <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", fontSize: "0.6rem", color: "#a855f7", fontWeight: 900 }}>NOW</div>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", color: "#666", fontSize: "0.6rem", fontWeight: 800, marginTop: 8 }}>
                    <span>LAP -4</span>
                    <span style={{ color: "#a855f7" }}>LAP SELECT</span>
                    <span>LAP +4</span>
                </div>
            </div>
        </div>
    );
}

