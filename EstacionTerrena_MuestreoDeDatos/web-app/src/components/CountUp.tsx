/* Número con count-up: useMotionValue + useSpring + useTransform (Motion). */
import { useEffect } from 'react'
import { motion, useMotionValue, useSpring, useTransform } from 'motion/react'

interface Props { value: number | null; decimals?: number; suffix?: string; fallback?: string }

export function CountUp({ value, decimals = 0, suffix = '', fallback = '—' }: Props) {
  const mv = useMotionValue(value ?? 0)
  const spring = useSpring(mv, { stiffness: 70, damping: 22, mass: 0.6 })
  const text = useTransform(spring, v => (value === null ? fallback : v.toFixed(decimals) + suffix))
  useEffect(() => { mv.set(value ?? 0) }, [value, mv])
  if (value === null) return <span>{fallback}</span>
  return <motion.span className="tabular-nums">{text}</motion.span>
}
