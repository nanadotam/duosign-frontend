import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-primary)] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-[var(--color-primary)] text-white hover:bg-[var(--color-primary-dark)] shadow-[var(--shadow-sm)] hover:shadow-[var(--shadow-md)] rounded-[var(--radius-md)]",
        cta: "bg-[var(--color-accent)] text-white hover:bg-[var(--color-accent-hover)] hover:-translate-y-[1px] active:translate-y-0 shadow-[var(--shadow-accent)] hover:shadow-[var(--shadow-lg)] rounded-[var(--radius-md)] font-semibold uppercase tracking-wider",
        secondary: "bg-[var(--panel-content-bg)] text-[var(--color-text-primary)] border border-[var(--panel-border)] hover:border-[var(--color-gray-300)] dark:hover:border-[var(--color-gray-600)] hover:shadow-[var(--shadow-sm)] rounded-[var(--radius-md)]",
        outline: "border border-[var(--panel-border)] bg-transparent hover:bg-[var(--panel-content-bg)] hover:border-[var(--color-gray-300)] dark:hover:border-[var(--color-gray-600)] rounded-[var(--radius-md)]",
        ghost: "hover:bg-[var(--panel-content-bg)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] rounded-[var(--radius-md)]",
        link: "text-[var(--color-primary)] underline-offset-4 hover:underline",
        playback: "text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--panel-content-bg)] rounded-[var(--radius-lg)]",
      },
      size: {
        default: "h-10 px-5 py-2",
        sm: "h-8 px-3 text-xs",
        lg: "h-12 px-8 py-3 text-base",
        icon: "h-10 w-10",
        playbackIcon: "h-11 w-11",
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
