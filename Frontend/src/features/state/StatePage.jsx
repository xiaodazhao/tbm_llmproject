import { useEffect, useState } from "react";
import api from "@/api/client";
import RockBand from "./RockBand";
export default function StatePage({ date }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    if (!date) return;

    setData(null);

    api
      .get(`/api/tbm/state?date=${date}`)
      .then((res) => {
        console.log("state api:", res.data); // ⭐调试用
        setData(res.data || {});
      })
      .catch((err) => {
        console.error("施工状态加载失败:", err);
        setData({});
      });
  }, [date]);

  if (!data) {
    return <div style={styles.loading}>正在加载施工状态…</div>;
  }

  const segments = Array.isArray(data.segments) ? data.segments : [];
  const efficiency = Array.isArray(data.efficiency) ? data.efficiency : [];
  const stateLabels = data.state_labels || {};

  return (
    <div style={styles.wrapper}>
      {/* ===== 时间轴 ===== */}
      <div style={styles.section}>
        <div style={styles.title}>🕒 状态时间轴</div>

        {segments.length === 0 ? (
          <div style={styles.empty}>暂无状态数据</div>
        ) : (
          <div style={styles.timeline}>
            {segments.map((s, i) => (
              <div key={i} style={styles.timelineItem}>
                <div style={styles.label}>
                  {s.label_text ||
                    stateLabels[s.label] ||
                    `状态 ${s.label}`}
                </div>

                <div style={styles.time}>
                  {safeTime(s.start)} ~ {safeTime(s.end)}
                </div>

                <div style={styles.duration}>
                  {formatDuration(s.duration)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ===== 效率统计 ===== */}
      <div style={styles.section}>
        <div style={styles.title}>📊 状态效率统计</div>

        {efficiency.length === 0 ? (
          <div style={styles.empty}>暂无效率数据</div>
        ) : (
          <div style={styles.tableContainer}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>状态</th>
                  <th style={styles.th}>平均推进速度</th>
                  <th style={styles.th}>平均推力</th>
                  <th style={styles.th}>平均扭矩</th>
                </tr>
              </thead>

              <tbody>
                {efficiency.map((row, i) => (
                  <tr key={i} style={styles.tr}>
                    <td style={styles.td}>
                      {row.label_text || "--"}
                    </td>

                    <td style={styles.td}>
                      {format(row["平均推进速度"])}
                    </td>

                    <td style={styles.td}>
                      {format(row["平均推力"])}
                    </td>

                    <td style={styles.td}>
                      {format(row["平均刀盘扭矩"])}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

/////////////////////// 工具函数 ///////////////////////

function format(v) {
  const num = Number(v);
  if (!isFinite(num)) return "--";
  return num.toFixed(2);
}

function formatDuration(v) {
  const num = Number(v);
  if (!isFinite(num)) return "--";
  return (num / 60).toFixed(1) + " min";
}

function safeTime(t) {
  if (!t) return "--";
  return t;
}

/////////////////////// 样式 ///////////////////////

const styles = {
  wrapper: {
    display: "flex",
    flexDirection: "column",
    gap: 16,
    height: "100%",
    padding: "4px",
  },

  section: {
    background: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: 12,
    padding: 16,
  },

  title: {
    fontSize: 15,
    fontWeight: 700,
    marginBottom: 12,
    color: "#1e293b",
  },

  timeline: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    maxHeight: 260,
    overflowY: "auto",
    paddingRight: "4px",
  },

  timelineItem: {
    padding: "10px 12px",
    borderRadius: 8,
    background: "#fff",
    border: "1px solid #e2e8f0",
    fontSize: 13,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },

  label: {
    fontWeight: 600,
    color: "#334155",
    flex: 1,
  },

  time: {
    color: "#64748b",
    flex: 2,
    textAlign: "center",
  },

  duration: {
    color: "#2563eb",
    fontWeight: 700,
    flex: 1,
    textAlign: "right",
  },

  tableContainer: {
    background: "#fff",
    borderRadius: 8,
    border: "1px solid #e2e8f0",
    overflow: "hidden",
  },

  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 13,
  },

  th: {
    background: "#f1f5f9",
    padding: "12px",
    textAlign: "left",
    color: "#475569",
    fontWeight: 700,
    borderBottom: "1px solid #e2e8f0",
  },

  td: {
    padding: "12px",
    color: "#334155",
    borderBottom: "1px solid #f1f5f9",
  },

  empty: {
    textAlign: "center",
    color: "#94a3b8",
    padding: 20,
  },

  loading: {
    textAlign: "center",
    color: "#64748b",
    paddingTop: "40px",
  },
};