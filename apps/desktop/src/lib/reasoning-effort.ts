import { normalize } from '@/lib/text'

/** Hermes' reasoning levels, in ascending order — mirrors the backend's
 *  VALID_REASONING_EFFORTS (hermes_constants.py). `none` is not a level: it's
 *  thinking disabled, owned by the Thinking toggle rather than the scale. */
export const REASONING_EFFORTS = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'] as const

export type ReasoningEffort = (typeof REASONING_EFFORTS)[number]

/** The scale plus the off state — the full set a config value may hold. */
export const REASONING_EFFORT_VALUES = ['none', ...REASONING_EFFORTS] as const

/** Hermes' built-in level when neither the surface nor the profile config
 *  specifies one (mirrors the backend's own fallback). */
export const DEFAULT_REASONING_EFFORT: ReasoningEffort = 'medium'

/** Compact labels for chrome where space is tight (pill, picker rows). Menus
 *  and settings use the translated `shell.modelOptions` strings instead. */
const SHORT_LABELS: Record<string, string> = {
  none: 'Off',
  minimal: 'Min',
  low: 'Low',
  medium: 'Med',
  high: 'High',
  xhigh: 'XHigh',
  max: 'Max',
  ultra: 'Ultra'
}

export function reasoningEffortLabel(effort: string): string {
  const key = normalize(effort)

  return key ? (SHORT_LABELS[key] ?? effort) : ''
}

export const isReasoningEffort = (value: string): value is ReasoningEffort =>
  REASONING_EFFORTS.includes(normalize(value) as ReasoningEffort)

/** Thinking is on unless a level explicitly says otherwise; an empty value
 *  means "inherit", so it resolves through `fallback` first. */
export const isThinkingEnabled = (effort: string, fallback: string = DEFAULT_REASONING_EFFORT): boolean =>
  normalize(effort || fallback) !== 'none'

/** Clamp a level onto a route's declared set, mirroring the backend's `clamp_effort`
 *  (agent/reasoning_effort.py): verbatim when supported, else the nearest WEAKER level so a
 *  clamp never escalates cost, else the weakest supported one. An undeclared set (`undefined`),
 *  an empty value (thinking off — the radio holds no selection) and a level outside the scale
 *  all pass through untouched, so this only narrows what a route actually constrains. */
export function clampReasoningEffort(effort: string, supported?: readonly string[]): string {
  if (!effort || supported === undefined) {
    return effort
  }

  const scale = REASONING_EFFORTS.filter(value => supported.includes(value))
  const index = REASONING_EFFORTS.indexOf(normalize(effort) as ReasoningEffort)

  if (!scale.length || index < 0 || scale.includes(REASONING_EFFORTS[index])) {
    return effort
  }

  const weaker = scale.filter(value => REASONING_EFFORTS.indexOf(value) < index)

  return weaker.length ? weaker[weaker.length - 1] : scale[0]
}

/** The level a scale control should show. Empty inherits `fallback`; `none`
 *  (thinking off) selects nothing; anything unrecognized clamps to the default. */
export function resolveReasoningEffort(effort: string, fallback: string = DEFAULT_REASONING_EFFORT): string {
  const value = normalize(effort || fallback)

  if (value === 'none') {
    return ''
  }

  return isReasoningEffort(value) ? value : DEFAULT_REASONING_EFFORT
}
