import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import api from "@/api/client";
export default function TimeWindowPage({ date }) {
  const [startTime, setStartTime] = useState("");
  const [endTime, setEndTime] = useState("");
  const [report, setReport] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(""); // 新增：专门用于显示错误信息

  // 🔄 当全局 date 变化时，重置时间窗
  useEffect(() => {
    if (date) {
      setStartTime(`${date}T08:00`);
      setEndTime(`${date}T12:00`);
      setReport("");
      setErrorMsg("");
    }
  }, [date]);

  const handleGenerate = async () => {
    // 1. 本地校验
    if (!startTime || !endTime) {
      setErrorMsg("⚠️ 请完整选择开始和结束时间");
      return;
    }
    if (startTime >= endTime) {
      setErrorMsg("⚠️ 结束时间必须晚于开始时间");
      return;
    }

    setLoading(true);
    setReport(""); // 清空旧内容
    setErrorMsg("");

    try {
      // 2. 发送请求
      // 后端接受 TimeWindowRequest: { start_time: str, end_time: str }
      // datetime-local 的格式 (YYYY-MM-DDTHH:mm) 可以直接发给后端
      // 后端会自动处理 "T" 的替换
      const res = await api.post("/api/tbm/report_by_time", { 
        start_time: startTime, 
        end_time: endTime 
      });

      // 3. 处理响应
      if (res.data.report && !res.data.report.startsWith("❌")) {
        setReport(res.data.report);
      } else {
        // 如果后端返回了包含错误符号的字符串
        setErrorMsg(res.data.report || "生成失败，未返回有效报告");
      }
    } catch (e) {
      console.error(e);
      setErrorMsg("❌ 网络请求出错或服务器异常");
    } finally {
      setLoading(false);
    }
  };

  // 如果没有选择日期，显示提示
  if (!date) {
    return <div style={{ padding: 20, color: '#64748b' }}>请先在左侧选择具体日期</div>;
  }

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div style={{ marginBottom: 20 }}>
         <h2 style={{ fontSize: "18px", color: "#1e293b", display: 'flex', alignItems: 'center', gap: '8px' }}>
           ⏱️ 时段分析 <span style={{fontSize: '14px', fontWeight: 'normal', color: '#64748b', background: '#f1f5f9', padding: '2px 8px', borderRadius: '4px'}}>{date}</span>
         </h2>
      </div>

      <div style={{ display: "flex", gap: 10, marginBottom: 20, background: "#f8fafc", padding: 15, borderRadius: 8, alignItems: 'flex-end' }}>
        <div style={{ flex: 1 }}>
            <label style={{display:'block', fontSize: 12, color: '#64748b', marginBottom: 4}}>开始时间</label>
            <input 
              type="datetime-local" 
              value={startTime} 
              // 关键修复：限制只能选当天，防止后端读错文件
              min={`${date}T00:00`}
              max={`${date}T23:59`}
              onChange={e => setStartTime(e.target.value)} 
              style={styles.input} 
            />
        </div>
        <div style={{ flex: 1 }}>
            <label style={{display:'block', fontSize: 12, color: '#64748b', marginBottom: 4}}>结束时间</label>
            <input 
              type="datetime-local" 
              value={endTime} 
              // 关键修复：限制只能选当天
              min={`${date}T00:00`}
              max={`${date}T23:59`}
              onChange={e => setEndTime(e.target.value)} 
              style={styles.input} 
            />
        </div>
        <button 
            onClick={handleGenerate} 
            disabled={loading} 
            style={{...styles.btn, opacity: loading ? 0.7 : 1}}
        >
            {loading ? "分析中..." : "生成报告"}
        </button>
      </div>

      {/* 错误提示区域 */}
      {errorMsg && (
        <div style={{ marginBottom: 20, padding: "10px", background: "#fee2e2", color: "#b91c1c", borderRadius: 8, fontSize: "14px" }}>
          {errorMsg}
        </div>
      )}

      <div style={{ flex: 1, overflowY: "auto", border: "1px dashed #cbd5e1", borderRadius: 8, padding: 20, background: '#fff' }}>
        {loading ? (
           <div style={{textAlign: 'center', color: '#94a3b8', marginTop: 50}}>
             ⏳ 正在分析数据并生成 AI 报告，请稍候...
           </div>
        ) : report ? (
           <div className="markdown-body">
             <ReactMarkdown>{report}</ReactMarkdown>
           </div>
        ) : (
           <div style={{color:'#94a3b8', textAlign:'center', marginTop: 50}}>
             请选择具体时间段并点击“生成报告”
           </div>
        )}
      </div>
    </div>
  );
}

const styles = {
    input: { 
      width: "100%", 
      padding: "8px", 
      border: "1px solid #cbd5e1", 
      borderRadius: 4, 
      fontFamily: 'inherit',
      color: '#334155'
    },
    btn: { 
      background: "#10b981", 
      color: "#fff", 
      border: "none", 
      padding: "0 20px", 
      height: "38px", // 对齐输入框高度
      borderRadius: 4, 
      cursor: "pointer", 
      fontWeight: "bold",
      minWidth: "100px"
    }
};