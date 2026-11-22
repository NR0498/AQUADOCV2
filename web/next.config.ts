import type { NextConfig } from "next";

// The Python API (FastAPI) runs separately. In development, proxy /api/* to it so the dashboard
// always talks to its own origin: no CORS, and no localhost-vs-127.0.0.1 surprises.
// On Vercel, the API is routed to /api by the project configuration instead.
const API_ORIGIN = process.env.AQUADOC_API_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    if (process.env.NODE_ENV !== "development") return [];
    return [{ source: "/api/:path*", destination: `${API_ORIGIN}/api/:path*` }];
  },
};

export default nextConfig;
