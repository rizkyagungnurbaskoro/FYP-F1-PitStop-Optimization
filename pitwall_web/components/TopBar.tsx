import Link from "next/link";

export default function TopBar() {
  return (
    <div className="topbar" style={{
      background: "rgba(16, 20, 28, 0.6)",
      backdropFilter: "blur(14px)",
      WebkitBackdropFilter: "blur(14px)",
      borderBottom: "1px solid rgba(255,255,255,0.08)",
      padding: "12px 24px",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      fontFamily: "var(--font-oxanium)",
      position: "sticky",
      top: 0,
      zIndex: 100
    }}>
      <div>
        <div className="brand" style={{ display: "flex", alignItems: "center", gap: 10, fontWeight: 900, letterSpacing: "2px", fontSize: "1.2rem" }}>
          <span className="brand-badge" style={{ background: "#E10600", padding: "2px 8px", borderRadius: 4, transform: "skew(-10deg)" }}>P1</span>
          PITWALL
        </div>
        <div className="tagline" style={{ fontSize: "0.7rem", color: "#17C3FF", textTransform: "uppercase", letterSpacing: "1px", fontWeight: 700 }}>Every Seconds Matters</div>
      </div>
      <nav className="nav" style={{ display: "flex", gap: 24 }}>
        <Link href="/overview" style={{ color: "#fff", textDecoration: "none", fontWeight: 600, fontSize: "0.9rem", textTransform: "uppercase", opacity: 0.8 }}>Overview</Link>
        <Link href="/demo" style={{ color: "#fff", textDecoration: "none", fontWeight: 600, fontSize: "0.9rem", textTransform: "uppercase", opacity: 0.8 }}>Iconic Scenario</Link>
        <Link href="/explore" style={{ color: "#fff", textDecoration: "none", fontWeight: 600, fontSize: "0.9rem", textTransform: "uppercase", opacity: 0.8 }}>Explore</Link>
      </nav>
    </div>
  );
}
