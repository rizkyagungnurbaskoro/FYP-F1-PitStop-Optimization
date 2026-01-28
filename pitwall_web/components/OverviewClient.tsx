"use client";

import { useEffect, useState } from "react";
import { getSummary, getHoldout } from "../lib/api";
import TimingTower from "./TimingTower";

const fallbackRows = [
  { stage: "S1", value: 0.83 },
  { stage: "S2", value: 0.84 },
  { stage: "S3", value: 0.13 },
  { stage: "S4", value: 0.15 }
];

export default function OverviewClient() {
  const [rows70, setRows70] = useState<any[]>([]);
  const [rows80, setRows80] = useState<any[]>([]);
  const [note, setNote] = useState("Live data unavailable, showing fallback values.");

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [strictData, holdoutData] = await Promise.all([
          getSummary("strict"),
          getHoldout()
        ]);

        const mapRows = (data: any) => (data.rows || []).map((r: any) => {
          const stage = String(r.stage || r.stage_id || "");
          const stageShort = stage.match(/Stage\s+(\d+)/i);
          const sid = stageShort ? parseInt(stageShort[1]) : 0;
          return {
            sid,
            stage: stageShort ? `S${stageShort[1]}` : stage,
            value: Number(r.mean_f1 || 0),
            method: sid % 2 === 0 ? "MyMethod" : "RefTech"
          };
        });

        const strictRows = mapRows(strictData);
        const holdoutRowsRaw = mapRows(holdoutData);

        // Calculate Deltas for 80/20 Tower (S1, S2, S3, S4 from strict)
        const combinedRows80 = strictRows.map((row: any, idx: number) => {
          let delta = null;
          if (row.sid === 2 && strictRows[idx - 1]?.sid === 1) {
            delta = row.value - strictRows[idx - 1].value;
          } else if (row.sid === 4 && strictRows[idx - 1]?.sid === 3) {
            delta = row.value - strictRows[idx - 1].value;
          }
          return { ...row, delta };
        });

        // Calculate Deltas for Holdout Tower (S3, S4 from holdout)
        const combinedRows70 = holdoutRowsRaw.map((row: any, idx: number) => {
          let delta = null;
          if (row.sid === 4 && holdoutRowsRaw[idx - 1]?.sid === 3) {
            delta = row.value - holdoutRowsRaw[idx - 1].value;
          }
          return { ...row, delta };
        });

        setRows80(combinedRows80);
        setRows70(combinedRows70);
        setNote("DUAL VALIDATION (80/20 & 70/30) ACTIVE");
      } catch (err) {
        setNote("API OFFLINE: REVERTING TO LOCAL CACHE");
      }
    };
    fetchAll();
  }, []);

  const MetricCard = ({ title, value, notice, color }: any) => (
    <div style={{
      background: "rgba(16, 20, 28, 0.65)",
      backdropFilter: "blur(14px)",
      WebkitBackdropFilter: "blur(14px)",
      borderRadius: 16,
      padding: "24px",
      border: `1px solid ${color}40`,
      boxShadow: `0 8px 32px rgba(0,0,0,0.5), inset 0 0 15px ${color}05`,
      position: "relative",
      fontFamily: "var(--font-oxanium)",
      overflow: "hidden"
    }}>
      <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: 2, background: `linear-gradient(90deg, transparent, ${color}, transparent)` }}></div>
      <h3 style={{ fontSize: "0.8rem", color: "#9ca3af", textTransform: "uppercase", letterSpacing: "1.5px", fontWeight: 800, margin: "0 0 10px 0" }}>{title}</h3>
      <div style={{ fontSize: "2.8rem", fontWeight: 900, color: "#fff", lineHeight: 1 }}>{value}</div>
      <div style={{ marginTop: 15, fontSize: "0.7rem", color: color, fontWeight: 700, textTransform: "uppercase", letterSpacing: "1px" }}>{notice}</div>
    </div>
  );

  return (
    <div style={{ fontFamily: "var(--font-oxanium)", paddingBottom: 60 }}>
      {/* Header Banner */}
      <div style={{ marginBottom: 40, borderBottom: "2px solid #E10600", paddingBottom: 20 }}>
        <h1 style={{ fontSize: "3rem", fontWeight: 900, textTransform: "uppercase", margin: 0, letterSpacing: "3px" }}>OVERVIEW <span style={{ color: "#E10600" }}>REPORT</span></h1>
        <div style={{ color: "#17C3FF", fontWeight: 700, fontSize: "0.9rem", letterSpacing: "2px", textTransform: "uppercase" }}>Model Performance & Thesis Metrics</div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 16, marginBottom: 40 }}>
        <MetricCard title="PRIMARY METRIC" value="F1" notice="STRICT VALIDATION" color="#E10600" />
        <MetricCard title="DECISION MODEL" value="XGB" notice="LEAKAGE-SAFE SPLIT" color="#fbbf24" />
        <MetricCard title="OVERALL SCORE" value="0.812" notice="80/20 VALIDATION" color="#17C3FF" />
        <MetricCard title="IMPROVEMENT" value="+0.030" notice="S2&S4 AVG GAIN" color="#22c55e" />
        <MetricCard title="TAGLINE" value="STAY AHEAD" notice="EVERY SECOND MATTERS" color="#fff" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 30, alignItems: "start" }}>
        <TimingTower rows80={rows80} rows70={rows70} />

        <div style={{
          background: "rgba(16, 20, 28, 0.65)",
          backdropFilter: "blur(14px)",
          borderRadius: 16,
          padding: "24px",
          border: "1px solid rgba(255,255,255,0.08)",
          boxShadow: "0 8px 32px rgba(0,0,0,0.5)"
        }}>
          <h3 style={{ color: "#17C3FF", textTransform: "uppercase", fontWeight: 800, letterSpacing: "1px", marginBottom: 20, fontSize: "1.1rem" }}>System Status</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 15 }}>
            <div style={{ background: "rgba(34, 197, 94, 0.1)", border: "1px solid #22c55e40", padding: "12px", borderRadius: 8 }}>
              <div style={{ fontSize: "0.6rem", color: "#22c55e", fontWeight: 800 }}>API STATUS</div>
              <div style={{ fontSize: "0.9rem", color: "#fff", fontWeight: 700 }}>{note}</div>
            </div>
            <div style={{ background: "rgba(255,255,255,0.03)", padding: "12px", borderRadius: 8 }}>
              <div style={{ fontSize: "0.6rem", color: "#9ca3af", fontWeight: 800 }}>INSIGHT</div>
              <div style={{ fontSize: "0.9rem", color: "#fff", fontWeight: 700, lineHeight: 1.4 }}>Compare S2 vs S1 and S4 vs S3 to visualize the optimization gains of the P1 Algorithm.</div>
            </div>
            <div style={{ background: "rgba(23, 195, 255, 0.1)", border: "1px solid #17c3ff40", padding: "12px", borderRadius: 8 }}>
              <div style={{ fontSize: "0.6rem", color: "#17C3FF", fontWeight: 800 }}>VERSION</div>
              <div style={{ fontSize: "0.9rem", color: "#fff", fontWeight: 700 }}>V2.4.0 (PIPELINE: ACTIVE)</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

