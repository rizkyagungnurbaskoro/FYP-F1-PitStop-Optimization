import type { Metadata } from "next";
import { Oxanium, Rajdhani } from "next/font/google";
import "./globals.css";

const rajdhani = Rajdhani({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-rajdhani",
  display: "swap"
});

const oxanium = Oxanium({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-oxanium",
  display: "swap"
});

export const metadata: Metadata = {
  title: "Pitwall Dashboard",
  description: "F1 pit-stop decision support dashboard"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${rajdhani.variable} ${oxanium.variable}`}>
        {children}
      </body>
    </html>
  );
}
