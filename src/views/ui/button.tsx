import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-primary-light)] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-[var(--color-primary)] text-white hover:bg-[var(--color-primary-dark)] rounded-[var(--radius-md)]",
        cta: "bg-[var(--color-cta)] text-white hover:bg-[var(--color-cta-hover)] active:bg-[var(--color-cta-active)] hover:-translate-y-[1px] active:translate-y-0 shadow-md hover:shadow-lg rounded-[var(--radius-md)] font-semibold uppercase tracking-wide",
        secondary: "bg-[var(--color-light-gray)] text-[var(--color-text-primary)] hover:bg-[var(--color-panel-gray)] rounded-[var(--radius-md)]",
        outline: "border border-[var(--color-mid-gray)] bg-transparent hover:bg-[var(--color-light-gray)] rounded-[var(--radius-md)]",
        ghost: "hover:bg-[var(--color-light-gray)] rounded-[var(--radius-md)]",
        link: "text-[var(--color-primary)] underline-offset-4 hover:underline",
        playback: "text-[var(--color-text-primary)] hover:bg-[var(--color-light-gray)] rounded-full",
      },
      size: {
        default: "h-11 px-6 py-2",
        sm: "h-9 px-4 text-xs",
        lg: "h-14 px-10 py-4 text-base",
        icon: "h-11 w-11",
        playbackIcon: "h-12 w-12",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
