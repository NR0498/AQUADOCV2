import { Suspense } from "react";
import StoredAnalysis from "@/components/StoredAnalysis";

// With Cache Components, URL params are runtime data: read them inside a Suspense boundary
// so the static shell (header, layout) can be prerendered and served instantly.
export default function AnalysisPage({ params }: PageProps<"/analyses/[id]">) {
  return (
    <Suspense fallback={<p className="p-6 text-sm text-ink-2">Loading analysis…</p>}>
      <StoredAnalysis params={params} />
    </Suspense>
  );
}
