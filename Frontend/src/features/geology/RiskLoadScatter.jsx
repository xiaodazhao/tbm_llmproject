import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ZAxis,
  Legend,
  Cell,
} from "recharts";

const RISK_COLORS = {
  high: "#ef4444",
  medium: "#f59e0b",
  low: "#10b981",
  none: "#94a3b8",
};

function safeNumber(v) {
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload || !payload.length) return null;

  const row = payload[0].payload || {};

  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        padding: 12,
        boxShadow: "0 8px 16px rgba(0,0,0,0.08)",
        minWidth: 220,
      }}
    >
      <div style={{ fontWeight: 700, color: "#0f172a", marginBottom: 8 }}>
        📍 {row.segment || "未知区段"}
      </div>

      <div style={{ fontSize: 13, color: "#475569", lineHeight: 1.8 }}>
        <div>
          风险等级：
          <span style={{ fontWeight: 700, color: RISK_COLORS[row.risk_mode] || "#94a3b8" }}>
            {row.risk_mode || "--"}
          </span>
        </div>
        <div>
          多源关注度：
          <span style={{ fontWeight: 700 }}>{row.active_source_count_max ?? "--"}</span>
        </div>
        <div>
          平均推力：
          <span style={{ fontWeight: 700 }}>{safeNumber(row["推力_mean"])?.toFixed(2) ?? "--"}</span>
        </div>
        <div>
          平均刀盘扭矩：
          <span style={{ fontWeight: 700 }}>{safeNumber(row["刀盘扭矩_mean"])?.toFixed(2) ?? "--"}</span>
        </div>
        <div>
          判读：
          <span style={{ fontWeight: 700 }}>{row.interpretation || "--"}</span>
        </div>
      </div>
    </div>
  );
};

export default function RiskLoadScatter({ data }) {
  const points = (data || [])
    .map((row) => ({
      ...row,
      x: safeNumber(row["推力_mean"]),
      y: safeNumber(row["刀盘扭矩_mean"]),
      z: Math.max(40, (Number(row.active_source_count_max || 1) || 1) * 30),
      fill: RISK_COLORS[row.risk_mode] || RISK_COLORS.none,
    }))
    .filter((row) => row.x !== null && row.y !== null);

  if (!points.length) {
    return (
      <div
        style={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#94a3b8",
          fontSize: 14,
        }}
      >
        当前缺少可用于风险-负载散点分析的数据
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" />

          <XAxis
            type="number"
            dataKey="x"
            name="平均推力"
            tick={{ fill: "#64748b", fontSize: 12 }}
            label={{
              value: "平均推力",
              position: "insideBottom",
              offset: -10,
              fill: "#475569",
            }}
          />

          <YAxis
            type="number"
            dataKey="y"
            name="平均刀盘扭矩"
            tick={{ fill: "#64748b", fontSize: 12 }}
            label={{
              value: "平均刀盘扭矩",
              angle: -90,
              position: "insideLeft",
              fill: "#475569",
            }}
          />

          <ZAxis type="number" dataKey="z" range={[60, 280]} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          <Scatter name="区段样本" data={points}>
            {points.map((entry, idx) => (
              <Cell key={idx} fill={entry.fill} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}