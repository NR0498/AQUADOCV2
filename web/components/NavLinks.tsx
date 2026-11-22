"use client";

import Link from "next/link";
import { Suspense } from "react";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Analyse" },
  { href: "/bands", label: "Band guide" },
  { href: "/model", label: "Methods & accuracy" },
];

function Links({ path }: { path: string | null }) {
  return (
    <nav className="order-last flex w-full gap-5 text-sm sm:order-none sm:ml-auto sm:w-auto">
      {NAV.map((item) => {
        const on = path !== null && (item.href === "/" ? path === "/" || path.startsWith("/analyses") : path.startsWith(item.href));
        return (
          <Link key={item.href} href={item.href} aria-current={on ? "page" : undefined}
            className={`no-underline hover:text-ink ${on ? "font-semibold text-ink" : "text-ink-2"}`}>
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

function ActiveLinks() {
  return <Links path={usePathname()} />;
}

/** The pathname is runtime data under Cache Components, so it streams in; the static shell shows plain links. */
export default function NavLinks() {
  return (
    <Suspense fallback={<Links path={null} />}>
      <ActiveLinks />
    </Suspense>
  );
}
