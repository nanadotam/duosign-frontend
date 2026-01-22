import type { Metadata } from "next";
import { Source_Serif_4 } from "next/font/google";
import "./globals.css";

const sourceSerif = Source_Serif_4({
  variable: "--font-serif",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "DuoSign - Visual-first Sign Language Translation",
  description: "Type text, see it signed. DuoSign is a Deaf-first visual translation workspace that converts typed text into sign language using 2D skeletal animation.",
  keywords: ["sign language", "translation", "accessibility", "deaf", "asl", "communication"],
  authors: [{ name: "DuoSign Team" }],
  openGraph: {
    title: "DuoSign - Visual-first Sign Language Translation",
    description: "Type text, see it signed. Make communication accessible for everyone.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${sourceSerif.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
