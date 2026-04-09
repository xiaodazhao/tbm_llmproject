import { useMemo, useState } from "react";

export default function RockBand({ segments, colorMap }) {
  if (!segments || segments.length === 0) {
    return (
      <div style={{ padding: 20, color: "#999", textAlign: "center" }}>
        暂无分段数据
      </div>
    );
  }

  const [hoverInfo, setHoverInfo] = useState(null);

  /* =========================
   *  颜色映射
   * ========================= */
  const defaultColors = {
    0: "#3b82f6",
    1: "#10b981",
    2: "#f59e0b",
    3: "#8b5cf6",
    4: "#ec4899",
  };
  const colors = colorMap || defaultColors;

  /* =========================
   *  工具函数
   * ========================= */
  const parseLabelId = (label) => {
    if (typeof label === "number") return label;
    if (typeof label === "string") {
      const m = label.match(/\d+/);
      return m ? Number(m[0]) : null;
    }
    return null;
  };

  const toSec = (t) => {
    if (!t) return 0;
    if (typeof t !== "string") t = String(t);
    const timePart = t.includes(" ") ? t.split(" ")[1] : t;
    const parts = timePart.split(":").map(Number);
    if (parts.length < 2 || parts.some((x) => Number.isNaN(x))) return 0;
    const [h, m, s = 0] = parts;
    return h * 3600 + m * 60 + s;
  };

  /* =========================
   *  排序与时间范围
   * ========================= */
  const sorted = useMemo(
    () => [...segments].sort((a, b) => toSec(a.start) - toSec(b.start)),
    [segments]
  );

  const allStartSecs = sorted.map((s) => toSec(s.start));
  const allEndSecs = sorted.map((s) => toSec(s.end));
  const minSec = Math.min(...allStartSecs);
  const maxSec = Math.max(...allEndSecs);
  const totalSec = maxSec - minSec || 1;

  /* =========================
   *  时间刻度生成（自动稀疏）
   * ========================= */
  const startHour = Math.floor(minSec / 3600);
  const endHour = Math.ceil(maxSec / 3600);
  const hourSpan = endHour - startHour;

  // 根据跨度自动决定刻度间隔
  let step = 1;
  if (hourSpan > 24) step = 4;
  else if (hourSpan > 12) step = 2;

  const ticks = [];
  for (let h = startHour; h <= endHour; h += step) {
    ticks.push(h);
  }

  return (
    <div style={{ padding: "10px 0", position: "relative", userSelect: "none" }}>
      {/* =========================
          时间刻度（斜排 + 稀疏）
         ========================= */}
      <div style={{ position: "relative", height: 15, marginBottom: 6 }}>
        {ticks.map((h) => {
          const pos = ((h * 3600 - minSec) / totalSec) * 100;
          if (pos < 0 || pos > 100) return null;

          return (
            <div
              key={h}
              style={{
                position: "absolute",
                left: `${pos}%`,
                transform: "translateX(-50%) rotate(-45deg)",
                transformOrigin: "top left",
                fontSize: 11,
                color: "#94a3b8",
                whiteSpace: "nowrap",
              }}
            >
              {String(h).padStart(2, "0")}:00
            </div>
          );
        })}
      </div>

      {/* =========================
          主时间轴
         ========================= */}
      <div
        style={{
          position: "relative",
          height: 48,
          borderRadius: 12,
          background: "#f1f5f9",
          overflow: "hidden",
          border: "1px solid #e2e8f0",
        }}
      >
        {sorted.map((s, i) => {
          const id = parseLabelId(s.label);
          const color = colors[id] || "#94a3b8";

          const segStartSec = toSec(s.start);
          const segEndSec = toSec(s.end);
          const left = ((segStartSec - minSec) / totalSec) * 100;
          let width = ((segEndSec - segStartSec) / totalSec) * 100;
          if (width <= 0) width = 0.5;

          return (
            <div
              key={i}
              style={{
                position: "absolute",
                left: `${left}%`,
                width: `${width}%`,
                top: 0,
                bottom: 0,
                background: color,
                borderRight: "2px solid #fff",
                cursor: "pointer",
              }}
              onMouseMove={(e) => {
                setHoverInfo({
                  x: e.clientX,
                  y: e.clientY,
                  id,
                  start: s.start,
                  end: s.end,
                  duration: (s.duration / 60).toFixed(1),
                  color,
                });
              }}
              onMouseLeave={() => setHoverInfo(null)}
            />
          );
        })}
      </div>

      {/* =========================
          图例
         ========================= */}
      <div style={{ marginTop: 14, display: "flex", gap: 18, flexWrap: "wrap" }}>
        {[...new Set(sorted.map((s) => parseLabelId(s.label)))]
          .filter((x) => x !== null)
          .sort()
          .map((id) => (
            <div
              key={id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 13,
                color: "#475569",
              }}
            >
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: "50%",
                  background: colors[id] || "#999",
                }}
              />
              施工状态 {id}
            </div>
          ))}
      </div>

      {/* =========================
          悬浮提示
         ========================= */}
      {hoverInfo && (
        <div
          style={{
            position: "fixed",
            left: hoverInfo.x + 16,
            top: hoverInfo.y + 16,
            padding: "10px 14px",
            background: "#fff",
            borderRadius: 8,
            boxShadow: "0 8px 16px rgba(0,0,0,0.12)",
            fontSize: 13,
            pointerEvents: "none",
            zIndex: 9999,
            minWidth: 150,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 6 }}>
            施工状态 {hoverInfo.id}
          </div>
          <div style={{ color: "#64748b" }}>
            {hoverInfo.start} ~ {hoverInfo.end}
          </div>
          <div style={{ marginTop: 4 }}>
            ⌛ {hoverInfo.duration} 分钟
          </div>
        </div>
      )}
    </div>
  );
}
