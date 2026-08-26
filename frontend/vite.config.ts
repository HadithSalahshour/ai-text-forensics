import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Loopback only, matching the backend. Never bind this to 0.0.0.0 —
    // it would expose every document you analyze to your local network.
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8420" },
  },
});
