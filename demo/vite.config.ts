import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

// public/models/<name>/ is the export.py output; the published page loads the same files.
export default defineConfig({
  base: process.env.PAGES_BASE ?? "/",
  root: fileURLToPath(new URL(".", import.meta.url)),
  publicDir: fileURLToPath(new URL("../public", import.meta.url)),
  server: { host: "127.0.0.1", port: 5174, allowedHosts: [".getbb.app"] },
  preview: { host: "127.0.0.1", port: 4174 },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { outDir: fileURLToPath(new URL("../dist-demo", import.meta.url)), target: "es2023", emptyOutDir: true },
});
