import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",  // 你的 FastAPI 地址
  timeout: 60000,
});

export default api;
