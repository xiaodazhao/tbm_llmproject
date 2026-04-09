export default function DateSelector({ dates, value, onChange }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
      <label style={{ fontWeight: "600", color: "#475569", fontSize: "14px" }}>
        📅 选择数据日期：
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          padding: "10px 16px",
          borderRadius: "8px",
          border: "1px solid #cbd5e1",
          fontSize: "16px",
          fontWeight: "bold",
          color: "#1e293b",
          cursor: "pointer",
          outline: "none",
          background: "#fff",
        }}
      >
        {dates.map((d) => (
          <option key={d} value={d}>
            {d}
          </option>
        ))}
      </select>
    </div>
  );
}