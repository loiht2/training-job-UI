const basePathEnv = process.env.NEXT_BASE_PATH;
const normalizedBasePath = basePathEnv && basePathEnv.trim() !== "" ? basePathEnv.replace(/\/$/, "") : "";

const nextConfig = {
  output: "standalone",
  ...(normalizedBasePath ? { basePath: normalizedBasePath } : {}),
};

export default nextConfig;
