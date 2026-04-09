import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
} from "recharts";
import api from "@/api/client";
/**
 * 里程格式化工具
 */
function formatChainage(chainage) {
  if (chainage === null || chainage === undefined || Number.isNaN(Number(chainage))) {
    return "";
  }
  const value = Number(chainage);
  const km = Math.floor(value / 1000);
  const m = value % 1000;
  return `DK${km}+${m.toFixed(2)}`;
}

export default function RiskProfilePage({ date }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!date) return;
    setData(null);
    setError(false);

    api
      .get(`/api/tbm/risk_profile?date=${date}`)
      .then((res) => {
        setData(res.data || {});
      })
      .catch((err) => {
        console.error("空间风险剖面加载失败:", err);
        setError(true);
      });
  }, [date]);

  const riskProfile = data?.risk_profile || {};
  const profile = riskProfile?.profile || [];
  const speedProfile = data?.speed_profile || [];

  // 合并风险数据与推进速度数据
  const mergedData = useMemo(() => {
    if (!profile?.length) return [];
    const speedMap = new Map();
    (speedProfile || []).forEach((s) => {
      speedMap.set(Number(s.chainage), Number(s["推进速度"]));
    });
    return profile.map((p) => ({
      chainage: Number(p.chainage),
      active_source_count: Number(p.active_source_count || 0),
      推进速度: speedMap.get(Number(p.chainage)) ?? null,
    }));
  }, [profile, speedProfile]);

  if (error) return <div style={styles.emptyBox}>❌ 空间风险剖面加载失败</div>;
  if (!data) return <div style={styles.emptyBox}>正在加载空间风险剖面…</div>;

  return (
    <div style={styles.wrapper}>
      {/* 顶部简要说明 */}
      <div style={styles.infoBox}>
        <div style={styles.infoTitle}>📈 风险-推进速度耦合分析</div>
        <div style={styles.infoText}>
          通过叠加地质风险（左轴）与掘进速度（右轴），识别高风险区间对施工进度的实际影响。
        </div>
      </div>

      {/* 耦合图表卡片 */}
      <div style={styles.chartCard}>
        <div style={styles.chartTitle}>隧道里程风险与推进速度耦合分布图</div>
        <div style={styles.chartInner}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mergedData} margin={{ top: 20, right: 60, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              
              <XAxis
                dataKey="chainage"
                tickFormatter={(v) => formatChainage(v)}
                tick={{ fontSize: 11, fill: "#64748b" }}
                minTickGap={60}
                label={{ value: "隧道里程 (DK标号)", position: "insideBottom", offset: -40, fill: "#475569", fontWeight: 600 }}
              />

              {/* 左侧 Y 轴：风险关注度 */}
              <YAxis
                yAxisId="left"
                tick={{ fontSize: 12, fill: "#64748b" }}
                allowDecimals={false}
                label={{ value: "风险关注度 (命中数)", angle: -90, position: "insideLeft", offset: 15, fill: "#dc2626" }}
              />

              {/* 右侧 Y 轴：推进速度 */}
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={{ fontSize: 12, fill: "#64748b" }}
                label={{ value: "推进速度 (mm/min)", angle: 90, position: "insideRight", offset: 15, fill: "#2563eb" }}
              />

              <Tooltip 
                labelFormatter={(label) => `里程位置：${formatChainage(label)}`}
                contentStyle={{ borderRadius: '10px', border: '1px solid #e2e8f0', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
              />
              
              <Legend 
                verticalAlign="top" 
                align="right" 
                height={50}
                iconType="rect"
              />

              {/* 高风险警戒线 */}
              <ReferenceLine 
                yAxisId="left"
                y={4} 
                stroke="#f59e0b" 
                strokeDasharray="5 5" 
                label={{ position: 'right', value: '关注阈值:4', fill: '#d97706', fontSize: 12, fontWeight: 700 }} 
              />

              {/* 风险曲线 */}
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="active_source_count" 
                stroke="#dc2626" 
                strokeWidth={3} 
                dot={false} 
                activeDot={{ r: 6 }}
                name="风险关注度" 
              />

              {/* 速度曲线 */}
              <Line 
                yAxisId="right"
                type="monotone" 
                dataKey="推进速度" 
                stroke="#2563eb" 
                strokeWidth={2} 
                strokeDasharray="5 2"
                dot={false} 
                name="平均推进速度" 
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

const styles = {
  wrapper: { 
    height: "100%", 
    display: "flex", 
    flexDirection: "column", 
    gap: 20, 
    padding: "20px",
    backgroundColor: "#f8fafc"
  },
  infoBox: { 
    background: "#fff", 
    border: "1px solid #e2e8f0", 
    borderRadius: 12, 
    padding: "16px 20px"
  },
  infoTitle: { fontSize: 16, fontWeight: 700, color: "#0f172a", marginBottom: 4 },
  infoText: { fontSize: 14, color: "#64748b" },
  chartCard: { 
    background: "#fff", 
    border: "1px solid #e2e8f0", 
    borderRadius: 16, 
    padding: "24px",
    boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
  },
  chartTitle: { fontSize: 15, fontWeight: 700, color: "#334155", marginBottom: 20, textAlign: "center" },
  chartInner: { width: "100%", height: 500 }, // 足够的高度确保不被挤压
  emptyBox: { height: 400, display: "flex", alignItems: "center", justifyContent: "center", color: "#94a3b8", fontSize: 15 }
};