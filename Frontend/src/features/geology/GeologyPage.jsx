import { useEffect, useMemo, useState } from "react";

import api from "@/api/client";
import GasChart from "@/features/gas/GasChart";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from "recharts";
function StatCard({ label, value, color = "#2563eb", bg = "#eff6ff" }) {
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
      <div
        style={{
          fontSize: 13,
          color: "#64748b",
          marginBottom: 10,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span>{label}</span>
        <span
          style={{
            width: 34,
            height: 34,
            borderRadius: 10,
            background: bg,
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            color,
            fontWeight: 700,
            fontSize: 14,
          }}
        >
          ●
        </span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color: "#0f172a", lineHeight: 1.1 }}>
        {value ?? 0}
      </div>
    </div>
  );
}

function formatNumber(v, digits = 2) {
  const num = Number(v);
  if (Number.isNaN(num)) return "--";
  return num.toFixed(digits);
}

function riskColor(risk) {
  const r = String(risk).toLowerCase();
  if (r === "high" || r === "高风险") return "#ef4444";
  if (r === "medium" || r === "中风险") return "#f59e0b";
  if (r === "low" || r === "低风险") return "#10b981";
  return "#94a3b8";
}

function riskBg(risk) {
  const r = String(risk).toLowerCase();
  if (r === "high" || r === "高风险") return "#fee2e2";
  if (r === "medium" || r === "中风险") return "#fef3c7";
  if (r === "low" || r === "低风险") return "#dcfce7";
  return "#f1f5f9";
}

export default function GeologyPage({ date }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!date) return;
    setData(null);
    setError(false);

    api
      .get(`/api/tbm/geology?date=${date}`)
      .then((res) => {
        setData(res.data || { record_summary: {}, segment_summary: {}, typical_segments: [] });
      })
      .catch((err) => {
        console.error("地质融合分析加载失败:", err);
        setError(true);
      });
  }, [date]);

  const recordSummary = data?.record_summary || {};
  const segmentSummary = data?.segment_summary || {};
  const typicalSegments = data?.typical_segments || [];

  // 构造柱状图数据
  const barData = useMemo(() => [
    { name: "低风险", value: recordSummary?.risk_counts?.low || 0, color: "#10b981" },
    { name: "中风险", value: recordSummary?.risk_counts?.medium || 0, color: "#f59e0b" },
    { name: "高风险", value: recordSummary?.risk_counts?.high || 0, color: "#ef4444" },
  ], [recordSummary]);

  const cards = useMemo(() => [
    { label: "高风险区段数", value: segmentSummary.high_risk_segment_count ?? 0, color: "#dc2626", bg: "#fee2e2" },
    { label: "多源关注区段数", value: segmentSummary.multi_source_segment_count ?? 0, color: "#d97706", bg: "#fef3c7" },
    { label: "典型区段数", value: typicalSegments.length ?? 0, color: "#2563eb", bg: "#dbeafe" },
  ], [segmentSummary, typicalSegments]);

  if (error) return <div style={styles.emptyBox}>❌ 地质融合分析数据加载失败</div>;
  if (!data) return <div style={styles.emptyBox}>正在加载地质融合分析结果…</div>;

  return (
    <div style={styles.wrapper}>
      {/* 1. 顶部指标卡 */}
      <div style={styles.cardGrid}>
        {cards.map((item, idx) => (
          <StatCard key={idx} label={item.label} value={item.value} color={item.color} bg={item.bg} />
        ))}
      </div>

      {/* 2. 区段级摘要 & 风险分布图 (左右布局) */}
      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 16 }}>
        <div style={styles.sectionCard}>
          <div style={styles.sectionTitle}>🧠 区段级地质融合摘要</div>
          <div style={styles.summaryText}>
            {segmentSummary.summary_text || "暂无区段级地质摘要。"}
          </div>
        </div>

        {/* 新加入的风险等级分布图 */}
        <div style={{ ...styles.sectionCard, background: "#fff" }}>
          <div style={styles.sectionTitle}>📊 风险等级分布 (记录级)</div>
          <div style={{ width: "100%", height: 220, marginTop: 10 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: "#64748b" }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: "#64748b" }} />
                <Tooltip 
                   cursor={{ fill: '#f8fafc' }}
                   contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]} barSize={40}>
                  {barData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 3. 记录级摘要 */}
      <details style={styles.sectionCard}>
        <summary style={{ ...styles.sectionTitle, cursor: "pointer" }}>📄 记录级摘要（点击展开）</summary>
        <div style={{ ...styles.summaryText, marginTop: 12 }}>
          {recordSummary.summary_text || "暂无记录级摘要。"}
        </div>
      </details>

      {/* 4. 典型区段表 */}
      <div style={{ ...styles.sectionCard, flex: 1, minHeight: 0 }}>
        <div style={styles.sectionTitle}>📍 典型区段识别结果</div>
        {typicalSegments.length === 0 ? (
          <div style={styles.emptyInner}>暂无典型区段数据</div>
        ) : (
          <div style={styles.tableWrap}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>区段</th>
                  <th style={styles.th}>风险等级</th>
                  <th style={styles.th}>多源关注</th>
                  <th style={styles.th}>平均推进速度</th>
                  <th style={styles.th}>平均推力</th>
                  <th style={styles.th}>结论</th>
                </tr>
              </thead>
              <tbody>
                {typicalSegments.map((row, idx) => {
                  const risk = row.risk_mode || row.risk || "";
                  return (
                    <tr key={idx}>
                      <td style={styles.td}>{row.segment || "--"}</td>
                      <td style={styles.td}>
                        <span style={{ display: "inline-block", padding: "2px 10px", borderRadius: 999, background: riskBg(risk), color: riskColor(risk), fontSize: 12, fontWeight: 700 }}>
                          {risk || "--"}
                        </span>
                      </td>
                      <td style={styles.td}>{row.active_source_count_max ?? "--"}</td>
                      <td style={styles.td}>{formatNumber(row["推进速度_mean"], 2)}</td>
                      <td style={styles.td}>{formatNumber(row["推力_mean"], 2)}</td>
                      <td style={{ ...styles.td, whiteSpace: 'normal', minWidth: 200 }}>{row.interpretation || "--"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  wrapper: { height: "100%", display: "flex", flexDirection: "column", gap: 16, paddingBottom: 20 },
  cardGrid: { display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 14 },
  sectionCard: { background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 14, padding: 16 },
  sectionTitle: { fontSize: 15, fontWeight: 700, color: "#334155" },
  summaryText: { fontSize: 14, color: "#475569", lineHeight: 1.8, whiteSpace: "pre-wrap" },
  tableWrap: { marginTop: 8, overflowX: "auto", overflowY: "auto", maxHeight: 350, borderRadius: 10, border: "1px solid #e2e8f0", background: "#fff" },
  table: { width: "100%", borderCollapse: "collapse" },
  th: { position: "sticky", top: 0, background: "#f1f5f9", padding: "10px 12px", textAlign: "left", fontSize: 13, zIndex: 1 },
  td: { fontSize: 13, color: "#475569", padding: "10px 12px", borderBottom: "1px solid #f1f5f9" },
  emptyBox: { height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#94a3b8" },
  emptyInner: { padding: 30, textAlign: "center", color: "#94a3b8" }
};