// Start the Python API and the dashboard together:  npm run dev:all
// Uses the repository's .venv (create it with: py -3.12 -m venv .venv && .venv\Scripts\pip install -e ".[dev]").
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const root = resolve(web, "..");
const python = process.platform === "win32" ? join(root, ".venv", "Scripts", "python.exe") : join(root, ".venv", "bin", "python");

if (!existsSync(python)) {
  console.error(`\nPython environment not found at ${python}\n` +
    "Create it from the repository root:\n" +
    "  py -3.12 -m venv .venv\n" +
    "  .venv\\Scripts\\python -m pip install -e \".[dev]\"\n");
  process.exit(1);
}

const children = [];
// On Windows, killing a child does not kill its children (npm -> next, python -> workers),
// which leaves ports 3000/8000 busy. Kill the whole process tree instead.
function stop(child) {
  if (child.exitCode !== null) return;
  if (process.platform === "win32") spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"]);
  else child.kill("SIGTERM");
}
function run(name, command, args, cwd, colour) {
  const child = spawn(command, args, { cwd, shell: process.platform === "win32" && command === "npm", env: process.env });
  const tag = `\x1b[${colour}m[${name}]\x1b[0m `;
  const write = (stream) => (chunk) => {
    for (const line of chunk.toString().split(/\r?\n/)) if (line.trim()) stream.write(tag + line + "\n");
  };
  child.stdout.on("data", write(process.stdout));
  child.stderr.on("data", write(process.stderr));
  child.on("exit", (code) => {
    console.log(`${tag}exited with code ${code}; stopping everything`);
    for (const other of children) if (other !== child) stop(other);
    setTimeout(() => process.exit(code ?? 0), 500); // let taskkill finish
  });
  children.push(child);
}

run("api", python, ["-m", "uvicorn", "aquadoc_core.api:app", "--port", "8000"], root, "36");
run("web", "npm", ["run", "dev"], web, "35");
console.log("\nDashboard: http://localhost:3000   API docs: http://localhost:3000/api/docs\n");
for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, () => { for (const child of children) stop(child); setTimeout(() => process.exit(0), 500); });
}
