import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import api from "@/api/client";
export default function ReportPage({ date }) {
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState("");
  const [error, setError] = useState("");

  // 日期变化时清空旧内容
  useEffect(() => {
    setReport("");
    setError("");
  }, [date]);

  const handleGenerate = async () => {
    if (!date) return;

    setLoading(true);
    setReport("");
    setError("");

    try {
      const res = await api.post("/api/tbm/report", { date });
      setReport(res.data.report);
    } catch (err) {
      console.error(err);
      setError("❌ 生成失败，请检查后端服务或稍后再试。");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* ===== 顶部 ===== */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 20,
        }}
      >
        <h2 style={{ margin: 0, fontSize: 18, color: "#1e293b" }}>
          📝 智能日报 ({date})
        </h2>

        <button
          onClick={handleGenerate}
          disabled={loading || !date}
          style={{
            ...styles.btn,
            opacity: loading || !date ? 0.6 : 1,
            cursor: loading || !date ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "正在分析..." : "生成日报"}
        </button>
      </div>

      {/* ===== 报告展示区 ===== */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          background: "#f8fafc",
          padding: 20,
          borderRadius: 8,
          border: "1px solid #e2e8f0",
        }}
      >
        {error && (
          <div
            style={{
              color: "#ef4444",
              background: "#fee2e2",
              padding: 10,
              borderRadius: 6,
              marginBottom: 10,
            }}
          >
            {error}
          </div>
        )}

        {report ? (
          <div
            className="markdown-body"
            style={{
              lineHeight: 1.6,
              color: "#334155",
              whiteSpace: "normal",     // ⭐ 关键：防止表格被 pre-wrap 搞乱
              wordBreak: "break-word",
            }}
          >
            <ReactMarkdown
              components={{
                table: ({ children, ...props }) => (
                  <div style={{ width: "100%", overflowX: "auto" }}>
                    <table
                      {...props}
                      style={{
                        width: "100%",
                        borderCollapse: "collapse",
                        tableLayout: "auto",
                        margin: "8px 0",
                      }}
                    >
                      {children}
                    </table>
                  </div>
                ),
                th: ({ children, ...props }) => (
                  <th
                    {...props}
                    style={{
                      border: "1px solid #e2e8f0",
                      background: "#f1f5f9",
                      padding: "8px 10px",
                      textAlign: "left",
                      fontWeight: 600,
                      whiteSpace: "nowrap",
                    }}
                  >
                    {children}
                  </th>
                ),
                td: ({ children, ...props }) => (
                  <td
                    {...props}
                    style={{
                      border: "1px solid #e2e8f0",
                      padding: "8px 10px",
                      verticalAlign: "top",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {children}
                  </td>
                ),
              }}
            >
              {report}
            </ReactMarkdown>
          </div>
        ) : (
          !loading && (
            <div
              style={{
                color: "#94a3b8",
                textAlign: "center",
                marginTop: 50,
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              <span style={{ fontSize: 24 }}>🤖</span>
              <span>点击上方按钮，AI 将为您生成 {date} 的工程日报</span>
            </div>
          )
        )}
      </div>
    </div>
  );
}

const styles = {
  btn: {
    background: "#3b82f6",
    color: "#fff",
    border: "none",
    padding: "8px 16px",
    borderRadius: 6,
    transition: "0.2s",
  },
};
// Below is partial code of c:\Users\22923\Desktop\LLM_20251219(1)\LLM_20251219\Frontend\node_modules\csstype\index.d.ts: