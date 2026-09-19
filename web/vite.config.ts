import { defineConfig } from "vite";

// base "./" so the bundle works from the server's static mount at "/"
// regardless of any deployment prefix. assetsInlineLimit 0 keeps the
// pdf.js worker a real file (the server CSP only allows 'self' workers).
export default defineConfig({
  base: "./",
  build: {
    outDir: "dist",
    assetsInlineLimit: 0,
    target: "es2022",
  },
});
