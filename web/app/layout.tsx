import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Newsreader } from "next/font/google";
import Link from "next/link";
import ApiStatus from "@/components/ApiStatus";
import NavLinks from "@/components/NavLinks";
import "./globals.css";

const newsreader = Newsreader({ subsets: ["latin"], variable: "--font-newsreader", weight: ["400", "500", "600"] });
const plexSans = IBM_Plex_Sans({ subsets: ["latin"], variable: "--font-plex-sans", weight: ["400", "500", "600"] });
const plexMono = IBM_Plex_Mono({ subsets: ["latin"], variable: "--font-plex-mono", weight: ["400", "500"] });

export const metadata: Metadata = {
  title: "AquaDoc — Satellite water-quality observatory",
  description:
    "Estimate algae, sediment, clarity and dissolved organic matter for any lake from Sentinel-2 imagery, and see exactly how each image was processed.",
};


export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${newsreader.variable} ${plexSans.variable} ${plexMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <div className="mx-auto w-full max-w-[1240px] px-4 sm:px-10">
          <header className="flex flex-wrap items-baseline gap-x-7 gap-y-2 border-b border-ink pb-4 pt-5">
            <Link href="/" className="font-serif text-[30px] font-semibold tracking-tight text-ink no-underline">
              Aqua<span className="text-accent">Doc</span>
            </Link>
            <span className="hidden font-mono text-[11px] uppercase tracking-[0.08em] text-muted md:inline">
              Satellite water-quality observatory
            </span>
            <NavLinks />
            <ApiStatus />
          </header>
          <div className="flex flex-wrap gap-x-5 gap-y-1 border-b border-hairline py-2 font-mono text-[11px] text-muted">
            <span>Sentinel-2 L2A · 20 m</span>
            <span>Model trained on GLORIA · validated on 506 satellite match-ups</span>
            <span>Research demonstration — not a safety assessment</span>
          </div>
        </div>
        <main className="flex-1">{children}</main>
        <footer className="mx-auto flex w-full max-w-[1240px] flex-wrap justify-between gap-3 px-4 pb-12 pt-7 text-xs text-muted sm:px-10">
          <span>Imagery: Copernicus Sentinel-2 via Microsoft Planetary Computer · Maps © OpenStreetMap contributors, OpenFreeMap</span>
          <span>Model trained on GLORIA (Lehmann et al. 2023, CC-BY 4.0)</span>
        </footer>
      </body>
    </html>
  );
}
