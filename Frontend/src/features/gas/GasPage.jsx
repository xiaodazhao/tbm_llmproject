import { useEffect, useMemo, useState } from "react";
import api from "../../api/client";
import GasChart from "./GasChart";

/* =========================
   通用统计卡片
   ========================= */
const StatCard = ({ label, value, unit = "", color = "#333" }) => {
  const display =
    typeof value === "number" && !Number.isNaN(value)
      ? value.toFixed(3)
      : "--";

  return (
    <div
      style={{
        background: "#f8fafc",
        padding: "12px",
        borderRadius: "8px",
        flex: 1,
        textAlign: "center",
        border: "1px solid #e2e8f0",
        minWidth: "80px",
      }}
    >
      <div style={{ color: "#64748b", fontSize: "12px", marginBottom: "4px" }}>
        {label}
      </div>
      <div style={{ color, fontSize: "18px", fontWeight: "bold" }}>
        {display}
        <span style={{ fontSize: "12px", marginLeft: "2px", color: "#94a3b8" }}>
          {unit}
        </span>
      </div>
    </div>
  );
};

const VIEW_OPTIONS = [
  { key: "all", label: "全天" },
  { key: "work", label: "掘进期" },
  { key: "stop", label: "停机期" },
];

const GAS_META = {
  CO2检测: { short: "CO₂", title: "二氧化碳", unit: "%" },
  H2S检测: { short: "H₂S", title: "硫化氢", unit: "" },
  SO2检测: { short: "SO₂", title: "二氧化硫", unit: "" },
  NO2检测: { short: "NO₂", title: "二氧化氮", unit: "" },
  NO检测: { short: "NO", title: "一氧化氮", unit: "" },
  CH4检测: { short: "CH₄", title: "甲烷", unit: "%" },
};

export default function GasPage({ date }) {
  const [gas, setGas] = useState(null);
  const [error, setError] = useState(false);
  const [view, setView] = useState("all");
  const [focusGas, setFocusGas] = useState("CH4检测");

  useEffect(() => {
    if (!date) return;

    setGas(null);
    setError(false);

    api
      .get(`/api/tbm/gas?date=${date}`)
      .then((res) => setGas(res.data || {}))
      .catch(() => setError(true));
  }, [date]);

  // ✅ 先给默认值，保证 hook 每次都执行
  const gasView = gas?.[view] || {};

  const availableGasKeys = useMemo(() => {
    return Object.keys(GAS_META).filter((k) => gasView[k]);
  }, [gasView]);

  useEffect(() => {
    if (!availableGasKeys.length) return;
    if (!gasView[focusGas]) {
      setFocusGas(availableGasKeys[0]);
    }
  }, [availableGasKeys, gasView, focusGas]);

  const focusMeta = GAS_META[focusGas] || {
    short: focusGas,
    title: focusGas,
    unit: "",
  };

  const focusData = gasView[focusGas] || {};

  const exceedCount =
    typeof focusData.exceed_event_count === "number"
      ? focusData.exceed_event_count
      : 0;

  const focusStatus = exceedCount > 0 ? "Abnormal" : "Normal";

  const badgeStyle =
    focusStatus === "Abnormal"
      ? { background: "#fee2e2", color: "#ef4444" }
      : { background: "#dcfce7", color: "#166534" };

  if (error) {
    return (
      <div style={{ padding: 40, color: "red" }}>
        ❌ 气体监测数据加载失败
      </div>
    );
  }

  if (!gas) {
    return (
      <div style={{ padding: 40, color: "#64748b" }}>
        正在加载气体监测数据…
      </div>
    );
  }

  return (
    <div
      style={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        gap: "20px",
      }}
    >
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {VIEW_OPTIONS.map((btn) => (
          <button
            key={btn.key}
            onClick={() => setView(btn.key)}
            style={{
              padding: "4px 10px",
              borderRadius: 6,
              border: "1px solid #e2e8f0",
              background: view === btn.key ? "#e0f2fe" : "#fff",
              color: view === btn.key ? "#0284c7" : "#475569",
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            {btn.label}
          </button>
        ))}
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {Object.entries(GAS_META).map(([key, meta]) => (
          <button
            key={key}
            onClick={() => setFocusGas(key)}
            style={{
              padding: "4px 10px",
              borderRadius: 6,
              border: "1px solid #e2e8f0",
              background: focusGas === key ? "#fef3c7" : "#fff",
              color: focusGas === key ? "#b45309" : "#475569",
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            {meta.short}
          </button>
        ))}
      </div>

      <div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: "10px",
          }}
        >
          <h3 style={{ margin: 0, fontSize: "15px", color: "#475569" }}>
            🔵 {focusMeta.title}（{focusMeta.short}）·{" "}
            {view === "all" ? "全天" : view === "work" ? "掘进期" : "停机期"}
          </h3>

          <span
            style={{
              ...badgeStyle,
              padding: "2px 8px",
              borderRadius: "12px",
              fontSize: "12px",
              fontWeight: "bold",
            }}
          >
            {focusStatus}
          </span>
        </div>

        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
          <StatCard
            label="平均值"
            value={focusData.mean}
            unit={focusMeta.unit}
            color="#3b82f6"
          />
          <StatCard label="最大值" value={focusData.max} unit={focusMeta.unit} />
          <StatCard label="最小值" value={focusData.min} unit={focusMeta.unit} />
        </div>

        {exceedCount > 0 && (
          <div style={{ marginTop: 8, fontSize: 12, color: "#ef4444" }}>
            ⚠ 发生 {exceedCount} 次超阈值事件
          </div>
        )}
      </div>

      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minHeight: 0,
        }}
      >
        <h3 style={{ marginBottom: 10, fontSize: "15px", color: "#475569" }}>
          📊 气体监测统计（
          {view === "all" ? "全天" : view === "work" ? "掘进期" : "停机期"}）
        </h3>

        <div
          style={{
            flex: 1,
            background: "#f8fafc",
            borderRadius: 12,
            border: "1px solid #e2e8f0",
            padding: 10,
          }}
        >
          <GasChart gasData={gasView} />
        </div>
      </div>
    </div>
  );
}