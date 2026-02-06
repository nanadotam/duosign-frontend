import type { Metadata } from "next";
import { Inter, Instrument_Serif } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const instrumentSerif = Instrument_Serif({
  variable: "--font-brand",
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
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
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                const theme = localStorage.getItem('duosign-theme');
                if (theme === 'dark') document.documentElement.classList.add('dark');
              } catch (e) {}
            `,
          }}
        />
      </head>
      <body className={`${inter.variable} ${instrumentSerif.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
