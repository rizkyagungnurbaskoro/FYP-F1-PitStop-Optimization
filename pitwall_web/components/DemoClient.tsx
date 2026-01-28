"use client";

import { useEffect, useState, useCallback } from "react";
import { getDemoContext, getDemoDecision, getDemoTelemetry, getDemoPitWindow, getDemoImpact } from "../lib/api";
import CallPanel from "./CallPanel";
import ContextChips from "./ContextChips";
import ImpactPanel from "./ImpactPanel";
import TimingTower from "./TimingTower";
import TelemetrySection from "./TelemetrySection";
import PitWindowSection from "./PitWindowSection";
import StrategySection from "./StrategySection";
import RaceControl from "./RaceControl";
import MonacoDeepDive from "./MonacoDeepDive";
import BroadcastHeader from "./BroadcastHeader";
import LiveTicker from "./LiveTicker";

interface Meta {
  drivers: string[];
  circuits: string[];
  weather_vals: string[];
  lap_min?: number;
  lap_max?: number;
}

export default function DemoClient() {
  // State
  const [loading, setLoading] = useState(true);
  const [meta, setMeta] = useState<Meta>({ drivers: [], circuits: [], weather_vals: [] });

  // Selection State
  const [dataset, setDataset] = useState<"my" | "ref">("my");
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

  // UI State
  const [showMonaco, setShowMonaco] = useState(false);


  // Initial Load & context refresh
  const fetchContext = useCallback(async (isInit = false) => {
    try {
      const res = await getDemoContext(dataset, isInit ? undefined : driver, isInit ? undefined : circuit, isInit ? undefined : weather);
      if (isInit) {
        setMeta({
          drivers: res.drivers || [],
          circuits: res.circuits || [],
          weather_vals: res.weather_vals || [],
          lap_min: res.lap_min,
          lap_max: res.lap_max
        });
        // Set defaults from selection if empty
        const sel = res.selection || {};
        if (sel.driver) setDriver(sel.driver);
        if (sel.circuit) setCircuit(sel.circuit);
        if (sel.weather) setWeather(sel.weather);
        if (sel.lap) setLap(Number(sel.lap));
      }
    } catch (e: any) {
      setError(e.message);
    }
  }, [dataset]);

  useEffect(() => {
    fetchContext(true);
  }, [fetchContext]);

  // Run Scenario
  const runScenario = async () => {
    setLoading(true);
    setError("");
    try {
      const sel = { dataset, driver, circuit, weather, lap };

      // Parallel fetch for speed
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

  // Auto-run on mount after context logic (simulated by small delay or effect dependency if we wanted live updates)
  // For now, user clicks RUN.

  return (
    <div style={{ maxWidth: 1400, margin: "0 auto", paddingBottom: 60 }}>
      <BroadcastHeader />
      <LiveTicker />

      {/* Hero Section for Presentation */}
      <div className="hero" style={{
        background: "linear-gradient(135deg, rgba(225,6,0,0.1) 0%, rgba(23,195,255,0.05) 50%, rgba(16,20,28,0.95) 100%)",
        border: "2px solid rgba(225, 6, 0, 0.3)",
        borderRadius: 16, padding: "40px 48px", marginBottom: 32,
        textAlign: "center", position: "relative", overflow: "hidden",
        boxShadow: "0 20px 60px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.05)"
      }}>
        {/* Animated Border Gradient */}
        <div style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 2,
          background: "linear-gradient(90deg, #E10600, #17C3FF, #FFD700, #E10600)",
          backgroundSize: "200% 100%",
          animation: "shimmer 3s linear infinite"
        }}></div>

        <div style={{ position: "relative", zIndex: 1 }}>
          <h1 style={{
            fontFamily: "var(--font-oxanium)", fontSize: "3rem", fontWeight: 900, letterSpacing: "0.08em",
            textTransform: "uppercase", marginBottom: 12,
            background: "linear-gradient(135deg, #fff, #17C3FF, #FFD700)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            textShadow: "0 0 40px rgba(225, 6, 0, 0.3)"
          }}>
            Pit Strategy Showcase
          </h1>
          <div style={{
            color: "#E10600",
            fontSize: "1.4rem",
            fontWeight: 800,
            textTransform: "uppercase",
            fontFamily: "var(--font-oxanium)",
            fontStyle: "italic",
            letterSpacing: "0.2em",
            marginBottom: 8,
            textShadow: "0 0 20px rgba(225, 6, 0, 0.5)"
          }}>
            MONACO 2022: FERRARI DOUBLE STACK STRATEGY
          </div>
          <div style={{ maxWidth: 700, margin: "20px auto 0", color: "#a6adbb", lineHeight: 1.6, fontSize: "1.05rem", fontFamily: "var(--font-oxanium)" }}>
            Analyzing Ferrari's catastrophic double-stack pit stop call during the 2022 Monaco Grand Prix.
            The ill-timed strategy cost Ferrari the race win, with Sainz dropping from the lead to P2.
            Evaluating the costly mistake through AI model analysis - was there a better strategy?
          </div>
        </div>

        {/* Background Decoration - Left */}
        <div style={{
          position: "absolute", top: -80, left: -80, width: 400, height: 400,
          background: "radial-gradient(circle, rgba(225,6,0,0.15) 0%, transparent 70%)",
          filter: "blur(60px)",
          pointerEvents: "none"
        }}></div>

        {/* Background Decoration - Right */}
        <div style={{
          position: "absolute", bottom: -80, right: -80, width: 400, height: 400,
          background: "radial-gradient(circle, rgba(23,195,255,0.15) 0%, transparent 70%)",
          filter: "blur(60px)",
          pointerEvents: "none"
        }}></div>
      </div>

      {/* Main Content: Monaco Deep Dive */}
      <MonacoDeepDive />

    </div>
  );
}

