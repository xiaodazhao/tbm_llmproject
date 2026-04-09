import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const SHORT_NAMES = {
  "CO2检测": "CO₂",
  "H2S检测": "H₂S",
  "SO2检测": "SO₂",
  "NO2检测": "NO₂",
  "NO检测": "NO",
  "CH4检测": "CH₄",
};

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div
        style={{
          backgroundColor: "rgba(255, 255, 255, 0.96)",
          border: "1px solid #ddd",
          padding: "10px",
          borderRadius: "6px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
          fontSize: "13px",
        }}
      >
        <p style={{ fontWeight: "bold", marginBottom: 6 }}>{label}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ color: entry.color, margin: 0 }}>
            {entry.name}: {Number(entry.value).toFixed(3)}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function GasChart({ gasData }) {
  const data = Object.keys(gasData || {})
    .filter((k) => gasData[k] && typeof gasData[k].max === "number")
    .map((k) => ({
      gas: SHORT_NAMES[k] ?? k,
      min: Math.max(0, Number(gasData[k].min ?? 0)),
      mean: Math.max(0, Number(gasData[k].mean ?? 0)),
      max: Math.max(0, Number(gasData[k].max ?? 0)),
    }))
    .filter((item) => item.max !== 0 || item.mean !== 0 || item.min !== 0);

  if (!data.length) {
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
        当前视图下暂无可展示的气体统计数据
      </div>
    );
  }

  return (
    <div style={{ height: 350 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} barGap={6} barCategoryGap={28}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />

          <XAxis
            dataKey="gas"
            tick={{ fill: "#666", fontSize: 12 }}
            axisLine={{ stroke: "#e0e0e0" }}
          />
          <YAxis
            tick={{ fill: "#999", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />

          <Tooltip content={<CustomTooltip />} cursor={{ fill: "transparent" }} />
          <Legend iconType="circle" wrapperStyle={{ paddingTop: "10px" }} />

          <Bar
            dataKey="min"
            fill="#34d399"
            name="最小值"
            radius={[4, 4, 0, 0]}
            animationDuration={900}
          />
          <Bar
            dataKey="mean"
            fill="#60a5fa"
            name="平均值"
            radius={[4, 4, 0, 0]}
            animationDuration={900}
          />
          <Bar
            dataKey="max"
            fill="#f87171"
            name="最大值"
            radius={[4, 4, 0, 0]}
            animationDuration={900}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}