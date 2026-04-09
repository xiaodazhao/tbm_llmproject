import { useEffect, useState } from "react";
import api from "@/api/client";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";

// 子组件：统计卡片
function Card({ title, value }) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 14,
        padding: 18,
        minHeight: 110,
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
      }}
    >
      <div style={{ fontSize: 13, color: "#64748b" }}>{title}</div>
      <div style={{ fontSize: 28, fontWeight: 800, color: "#0f172a" }}>
        {value ?? 0}
      </div>
    </div>
  );
}

export default function SummaryPage({ date }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    if (!date) return;
    setData(null);
    api
      .get(`/api/tbm/summary?date=${date}`)
      .then((res) => setData(res.data || {}))
      .catch((err) => {
        console.error("概览加载失败:", err);
        setData({});
      });
  }, [date]);

  if (!data) {
    return <div style={styles.loading}>正在加载概览…</div>;
  }

  // 准备饼图数据，确保 value 为数字
  const pieData = [
    { name: "掘进", value: Number(data.work_total_min || 0) },
    { name: "停机", value: Number(data.stop_total_min || 0) },
    { name: "过渡", value: Number(data.transition_total_min || 0) },
    { name: "异常", value: Number(data.abnormal_total_min || 0) },
  ];

  return (
    <div style={styles.wrapper}>
      {/* ===== 第一组：施工统计 ===== */}
      <section>
        <div style={styles.sectionTitle}>⚙️ 施工工况统计</div>
        <div style={styles.grid}>
          <Card title="稳定掘进段数" value={data.work_count} />
          <Card title="停机段数" value={data.stop_count} />
          <Card title="稳定掘进总时长 (min)" value={round(data.work_total_min)} />
          <Card title="停机总时长 (min)" value={round(data.stop_total_min)} />
        </div>
      </section>

      {/* ===== 第二组：地质风险概览 ===== */}
      <section>
        <div style={styles.sectionTitle}>🪨 地质风险概览</div>
        <div style={styles.grid}>
          <Card title="高风险区段数" value={data.geology_high_risk_segment_count} />
          <Card title="多源关注区段数" value={data.geology_multi_source_segment_count} />
          <Card title="异常工况段数" value={data.abnormal_count} />
          <Card title="过渡段数" value={data.transition_count} />
        </div>
      </section>

      {/* ===== 第三组：加入饼图展示 ===== */}
      <section style={styles.chartSection}>
        <div style={styles.sectionTitle}>⏱️ 工况时间占比</div>
        <div style={styles.chartContainer}>
          <div style={{ width: "100%", height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  innerRadius={60} // 改成环形图，更高级
                  paddingAngle={5}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  <Cell fill="#16a34a" /> {/* 掘进 */}
                  <Cell fill="#dc2626" /> {/* 停机 */}
                  <Cell fill="#0ea5e9" /> {/* 过渡 */}
                  <Cell fill="#7c3aed" /> {/* 异常 */}
                </Pie>
                <Tooltip />
                <Legend verticalAlign="bottom" height={36} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </div>
  );
}

function round(v) {
  const num = Number(v);
  return isNaN(num) ? "--" : num.toFixed(1);
}

const styles = {
  wrapper: {
    display: "flex",
    flexDirection: "column",
    gap: 24,
    height: "100%",
    paddingBottom: 40,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: 700,
    color: "#334155",
    marginBottom: 12,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
    gap: 14,
  },
  chartSection: {
    marginTop: 10,
  },
  chartContainer: {
    background: "#fff",
    border: "1px solid #e2e8f0",
    borderRadius: 14,
    padding: 20,
    boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
  },
  loading: {
    height: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#94a3b8",
  },
};