export default function SettingsPage() {
  return (
    <div style={{ marginLeft: 260, padding: "40px" }}>
      <h1 style={{ marginBottom: 20 }}>⚙️ 系统设置</h1>

      <div
        style={{
          background: "var(--card-bg)",
          padding: 24,
          borderRadius: 12,
          border: "1px solid var(--border)",
          maxWidth: 500,
          lineHeight: 1.8,
        }}
      >
        <h2 style={{ fontSize: 20, marginBottom: 10 }}>全局参数设置</h2>

        <p style={{ opacity: 0.7 }}>（暂未开放，可根据需求扩展）</p>

        <ul>
          <li>默认 CSV 路径设置</li>
          <li>LLM API 配置</li>
          <li>掘进阈值参数</li>
          <li>图表显示选项</li>
        </ul>
      </div>
    </div>
  );
}
