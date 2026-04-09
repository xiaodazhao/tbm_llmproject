import api from "./client";

export const tbmApi = {
  getDates: () => api.get("/api/tbm/dates"),
  getSummary: (date) => api.get(`/api/tbm/summary?date=${date}`),
  getGeology: (date) => api.get(`/api/tbm/geology?date=${date}`),
  getState: (date) => api.get(`/api/tbm/state?date=${date}`),
  getGas: (date) => api.get(`/api/tbm/gas?date=${date}`),
  getRiskProfile: (date) => api.get(`/api/tbm/risk_profile?date=${date}`),
  generateReport: (date) => api.post("/api/tbm/report", { date }),
  generateTimeWindowReport: (payload) =>
    api.post("/api/tbm/report_by_time", payload),
};