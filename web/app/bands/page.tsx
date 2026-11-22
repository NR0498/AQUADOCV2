import type { Metadata } from "next";
import BandGuide from "@/components/BandGuide";

export const metadata: Metadata = {
  title: "Band guide · AquaDoc",
  description: "What each Sentinel-2 band reveals about water, and which pollutants it helps estimate.",
};

export default function BandsPage() {
  return <BandGuide />;
}
