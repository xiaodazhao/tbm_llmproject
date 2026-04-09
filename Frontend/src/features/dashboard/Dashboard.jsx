import { useState, useEffect } from "react";

import api from "@/api/client";
import SummaryPage from "@/features/summary/SummaryPage";
import GeologyPage from "@/features/geology/GeologyPage";
import StatePage from "@/features/state/StatePage";
import GasPage from "@/features/gas/GasPage";
import ReportPage from "@/features/report/ReportPage";
import TimeWindowPage from "@/features/report/TimeWindowPage";
import RiskProfilePage from "@/features/risk/RiskProfilePage";

export default function Dashboard() {
  const [dates, setDates] = useState([]);
  const [currentDate, setCurrentDate] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDates = async () => {
      try {
        const res = await api.get("/api/tbm/dates");
        const list = res.data.dates || [];
        setDates(list);

        if (list.length > 0) {
          setCurrentDate(list[0]);
        }
      } catch (err) {
        console.error("加载日期失败", err);
      } finally {
        setLoading(false);
      }
    };

    fetchDates();
  }, []);

  if (loading) {
    return <div style={styles.loading}>🚀 系统初始化中...</div>;
  }

  return (
    <div style={styles.container}>
      {/* 顶部栏 */}
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>🏗️ TBM 智能监控驾驶舱</h1>
          <p style={styles.subtitle}>当前分析日期：{currentDate || "未选择"}</p>
        </div>

        <div style={styles.selectorWrapper}>
          <label style={styles.label}>📅 选择数据日期：</label>
          <select
            value={currentDate}
            onChange={(e) => setCurrentDate(e.target.value)}
            style={styles.select}
          >
            {dates.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>
      </header>

      {/* 主体网格 */}
      <div style={styles.grid}>
        {/* 1. 工况概览 */}
        <div style={{ ...styles.card, gridColumn: "span 2", minHeight: "420px" }}>
          <h2 style={styles.cardTitle}>📊 工况概览</h2>
          <SummaryPage date={currentDate} />
        </div>

        {/* 2. 地质融合分析 */}
        <div style={{ ...styles.card, gridColumn: "span 2", minHeight: "520px" }}>
          <h2 style={styles.cardTitle}>🪨 地质融合分析</h2>
          <GeologyPage date={currentDate} />
        </div>

        {/* 3. 施工状态分析 */}
        <div style={{ ...styles.card, gridColumn: "span 2", minHeight: "520px" }}>
          <h2 style={styles.cardTitle}>⚙️ 施工状态分析</h2>
          <StatePage date={currentDate} />
        </div>

        {/* 4. 气体监测 */}
        <div style={{ ...styles.card, gridColumn: "span 3", minHeight: "420px" }}>
          <h2 style={styles.cardTitle}>🌫 气体监测</h2>
          <GasPage date={currentDate} />
        </div>

        {/* 5. 空间风险剖面 */}
        <div style={{ ...styles.card, gridColumn: "span 3", minHeight: "420px" }}>
          <h2 style={styles.cardTitle}>🗺️ 空间风险剖面</h2>
          <RiskProfilePage date={currentDate} />
        </div>

        {/* 6. 智能日报 */}
        <div style={{ ...styles.card, gridColumn: "span 3", height: "680px" }}>
          <ReportPage date={currentDate} />
        </div>

        {/* 7. 时间窗分析 */}
        <div style={{ ...styles.card, gridColumn: "span 3", height: "680px" }}>
          <TimeWindowPage date={currentDate} />
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    padding: "40px",
    backgroundColor: "#f1f5f9",
    minHeight: "100vh",
  },

  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "30px",
    background: "#fff",
    padding: "20px 24px",
    borderRadius: "16px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.05)",
    border: "1px solid #e2e8f0",
  },

  title: {
    fontSize: "24px",
    fontWeight: "800",
    color: "#1e293b",
    margin: 0,
  },

  subtitle: {
    color: "#64748b",
    margin: 0,
    fontSize: "14px",
    marginTop: "4px",
  },

  selectorWrapper: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    flexWrap: "wrap",
  },

  label: {
    fontWeight: "600",
    color: "#475569",
    fontSize: "14px",
  },

  select: {
    padding: "10px 16px",
    borderRadius: "8px",
    border: "1px solid #cbd5e1",
    fontSize: "16px",
    fontWeight: "bold",
    color: "#1e293b",
    cursor: "pointer",
    outline: "none",
    background: "#fff",
  },

  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(6, 1fr)",
    gap: "24px",
    alignItems: "stretch",
  },

  card: {
    background: "#ffffff",
    borderRadius: "16px",
    border: "1px solid #e2e8f0",
    padding: "24px",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)",
  },

  cardTitle: {
    fontSize: "16px",
    fontWeight: "600",
    color: "#334155",
    marginBottom: "16px",
    borderBottom: "1px solid #f1f5f9",
    paddingBottom: "10px",
    flexShrink: 0,
  },

  loading: {
    height: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "20px",
    color: "#64748b",
    background: "#f8fafc",
  },
};