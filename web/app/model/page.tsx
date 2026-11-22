import type { Metadata } from "next";
import ModelCard from "@/components/ModelCard";

export const metadata: Metadata = {
  title: "Model & accuracy · AquaDoc",
  description: "How the water-quality model was trained and how accurate it is on real Sentinel-2 images.",
};

export default function ModelPage() {
  return <ModelCard />;
}
