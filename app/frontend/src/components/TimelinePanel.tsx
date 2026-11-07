import { useMemo, useState } from 'react'

export interface TimelineWindow {
  index: number
  start_frame: number
  end_frame: number
  start_s: number | null
  end_s: number | null
  top1: { label: string; score: number; class_index: number }
  topk: { rank: number; label: string; score: number }[]
  inference_s: number | null
}

export interface TimelineData {
  windows: TimelineWindow[]
  window_frames: number
  stride_frames: number
  window_seconds: number | null
  num_windows: number
  frames_covered: number
  duration_s: number
  overall_topk: { rank: number; label: string; score: number }[]
  label_votes: { label: string; windows_won: number; share: number }[]
}

// Stable colour per label so the same action keeps its colour down the strip.
function colourFor(label: string): string {
  let h = 0
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) % 360
  return `hsl(${h} 62% 52%)`
}

function fmt(s: number | null): string {
  if (s == null) return '—'
  const m = Math.floor(s / 60)
  const sec = s - m * 60
  return m > 0 ? `${m}:${sec.toFixed(1).padStart(4, '0')}` : `${sec.toFixed(1)}s`
}

export function TimelinePanel({ timeline }: { timeline: TimelineData }) {
  const [minScore, setMinScore] = useState(0)
  const [selected, setSelected] = useState<TimelineWindow | null>(null)

  const total = timeline.windows.length
  const shown = useMemo(
    () => timeline.windows.filter((w) => w.top1.score >= minScore),
    [timeline.windows, minScore],
  )

  // Merge consecutive windows with the same label into readable segments —
  // "0.0s–4.5s: walking" reads better than nine separate half-second rows.
  const segments = useMemo(() => {
    const out: { label: string; start: number | null; end: number | null; n: number; best: number }[] = []
    for (const w of timeline.windows) {
      const last = out[out.length - 1]
      if (last && last.label === w.top1.label) {
        last.end = w.end_s
        last.n += 1
        last.best = Math.max(last.best, w.top1.score)
      } else {
        out.push({ label: w.top1.label, start: w.start_s, end: w.end_s, n: 1, best: w.top1.score })
      }
    }
    return out
  }, [timeline.windows])

  const lastEnd = timeline.windows[timeline.windows.length - 1]?.end_s ?? 1

  return (
    <section className="panel">
      <h2>Timeline — what happened, second by second</h2>

      <div className="grid2" style={{ marginBottom: 14 }}>
        <dl className="stats">
          <dt>Windows analysed</dt>
          <dd>{timeline.num_windows}</dd>
          <dt>Window length</dt>
          <dd>
            {timeline.window_frames} frames
            {timeline.window_seconds ? ` (${timeline.window_seconds}s)` : ''}
          </dd>
          <dt>Step between windows</dt>
          <dd>{timeline.stride_frames} frames</dd>
        </dl>
        <dl className="stats">
          <dt>Frames covered</dt>
          <dd>
            {timeline.frames_covered} <span className="muted">(every frame)</span>
          </dd>
          <dt>Classification time</dt>
          <dd>{timeline.duration_s}s total</dd>
          <dt>Per window</dt>
          <dd>{(timeline.duration_s / Math.max(total, 1)).toFixed(3)}s</dd>
        </dl>
      </div>

      {/* ---- colour strip: the whole clip at a glance ---- */}
      <div className="small muted" style={{ marginBottom: 6 }}>
        Whole clip (hover a band for details)
      </div>
      <div
        style={{
          display: 'flex',
          height: 34,
          borderRadius: 6,
          overflow: 'hidden',
          border: '1px solid var(--border)',
          marginBottom: 6,
        }}
      >
        {timeline.windows.map((w) => (
          <div
            key={w.index}
            title={`${fmt(w.start_s)} – ${fmt(w.end_s)}\n${w.top1.label}\n${(w.top1.score * 100).toFixed(1)}%`}
            onClick={() => setSelected(w)}
            style={{
              flex: 1,
              background: colourFor(w.top1.label),
              // Confidence shown as opacity: faint band = the model is unsure.
              opacity: 0.35 + 0.65 * w.top1.score,
              cursor: 'pointer',
              borderRight: '1px solid rgba(0,0,0,0.25)',
            }}
          />
        ))}
      </div>
      <div className="row small muted" style={{ justifyContent: 'space-between', marginBottom: 16 }}>
        <span>0s</span>
        <span>{fmt(lastEnd)}</span>
      </div>

      {selected && (
        <div className="pred-top" style={{ marginBottom: 16 }}>
          <div className="small muted">
            {fmt(selected.start_s)} – {fmt(selected.end_s)} · frames {selected.start_frame}–
            {selected.end_frame}
          </div>
          <div className="pred-label">{selected.top1.label}</div>
          <div className="pred-conf">{(selected.top1.score * 100).toFixed(1)}%</div>
          <div style={{ marginTop: 10 }}>
            {selected.topk.slice(0, 5).map((t) => (
              <div className="bar-row" key={t.rank}>
                <span className="bar-rank">{t.rank}.</span>
                <span className="bar-track">
                  <span className="bar-fill" style={{ width: `${t.score * 100}%` }} />
                  <span className="bar-name">{t.label}</span>
                </span>
                <span className="bar-val">{t.score.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ---- merged segments ---- */}
      <details open>
        <summary>Segments ({segments.length})</summary>
        <table className="kv" style={{ marginTop: 8 }}>
          <thead>
            <tr>
              <th>From</th>
              <th>To</th>
              <th>Action</th>
              <th>Best conf.</th>
            </tr>
          </thead>
          <tbody>
            {segments.map((sg, i) => (
              <tr key={i}>
                <td className="mono">{fmt(sg.start)}</td>
                <td className="mono">{fmt(sg.end)}</td>
                <td>
                  <span
                    style={{
                      display: 'inline-block',
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      background: colourFor(sg.label),
                      marginRight: 8,
                    }}
                  />
                  {sg.label}
                </td>
                <td className="mono">{(sg.best * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      {/* ---- per-window list ---- */}
      <details style={{ marginTop: 12 }}>
        <summary>
          Every window ({total}) — filter by confidence
        </summary>
        <div className="row" style={{ margin: '10px 0' }}>
          <input
            type="range"
            min={0}
            max={0.95}
            step={0.05}
            value={minScore}
            onChange={(e) => setMinScore(parseFloat(e.target.value))}
            style={{ flex: 1 }}
          />
          <span className="mono small">
            ≥ {(minScore * 100).toFixed(0)}% · {shown.length}/{total}
          </span>
        </div>
        <table className="kv">
          <thead>
            <tr>
              <th>Time</th>
              <th>Frames</th>
              <th>Action</th>
              <th>Conf.</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((w) => (
              <tr key={w.index} onClick={() => setSelected(w)} style={{ cursor: 'pointer' }}>
                <td className="mono">
                  {fmt(w.start_s)} – {fmt(w.end_s)}
                </td>
                <td className="mono small">
                  {w.start_frame}–{w.end_frame}
                </td>
                <td>{w.top1.label}</td>
                <td className="mono">{(w.top1.score * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      {/* ---- which labels dominated ---- */}
      <div style={{ marginTop: 16 }}>
        <div className="small muted" style={{ marginBottom: 8 }}>
          Which action won the most windows
        </div>
        {timeline.label_votes.map((v) => (
          <div className="bar-row" key={v.label}>
            <span className="bar-rank" />
            <span className="bar-track">
              <span
                className="bar-fill"
                style={{ width: `${v.share * 100}%`, background: colourFor(v.label) }}
              />
              <span className="bar-name">{v.label}</span>
            </span>
            <span className="bar-val">
              {v.windows_won} · {(v.share * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>

      <div className="notice warn small" style={{ marginTop: 14 }}>
        The <strong>times</strong> are exact and the <strong>people</strong> are now the
        moving athletes — but the <strong>labels</strong> come from NTU RGB+D 60, which has
        no karate classes. The model must answer with one of its 60 everyday actions, so
        it cannot name a technique. Training on labelled karate footage is the only fix.
      </div>
    </section>
  )
}
