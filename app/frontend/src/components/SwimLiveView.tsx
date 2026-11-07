import { useCallback, useEffect, useRef, useState } from 'react'
import { SkeletonCanvas, type Keypoint, type LivePlayer } from './SkeletonCanvas'
import { STROKE_INFO } from './SwimResult'

interface Pred {
  label: string
  score: number
}

interface SwimSnap {
  running: boolean
  sport?: string
  frames_processed: number
  people_now: number
  buffer_fill: number
  window: number
  window_seconds?: number
  inference_s: number | null
  error: string | null
  frame_hw?: [number, number] | null
  latest_frame_jpeg?: string | null
  latest_keypoints?: Keypoint[][] | null
  frame_keypoints?: Keypoint[][] | null
  frame_players?: LivePlayer[] | null
  latest_players?: LivePlayer[] | null
  status?: 'warming' | 'no_person' | 'arms_not_visible' | 'not_swimming' | 'uncertain' | 'ok'
  arm_visible_share?: number
  arm_travel?: number
  lag_ms?: number | null
  stroke?: string | null
  confidence?: number | null
  standing_mode?: boolean
  prediction: { top1: Pred | null; topk: Pred[] } | null
}

const info = (k: string) => STROKE_INFO[k] ?? { name: k, ar: '', color: '#888' }

/** Live stroke recognition for the swimming page (separate from karate live). */
/** modelReady: null while the backend status is still loading. */
export function SwimLiveView({
  modelReady,
  engine,
  onRunningChange,
}: {
  modelReady: boolean | null
  /** pose model chosen at the top of the page (null = not chosen yet) */
  engine: string | null
  onRunningChange?: (running: boolean) => void
}) {
  const [snap, setSnap] = useState<SwimSnap | null>(null)
  const [starting, setStarting] = useState(false)
  const [standing, setStanding] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const ws = useRef<WebSocket | null>(null)

  const closeWs = useCallback(() => {
    ws.current?.close()
    ws.current = null
  }, [])
  useEffect(() => closeWs, [closeWs])

  const openWs = useCallback(() => {
    if (ws.current) return
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const sock = new WebSocket(`${proto}://${location.host}/api/live/ws`)
    sock.onmessage = (ev) => {
      const data = JSON.parse(ev.data) as SwimSnap
      setSnap((prev) => ({
        ...data,
        latest_frame_jpeg: data.latest_frame_jpeg ?? prev?.latest_frame_jpeg ?? null,
        frame_keypoints: data.latest_frame_jpeg
          ? data.frame_keypoints ?? null
          : prev?.frame_keypoints ?? data.frame_keypoints ?? null,
        frame_players: data.latest_frame_jpeg
          ? data.frame_players ?? null
          : prev?.frame_players ?? data.frame_players ?? null,
        frame_hw: data.frame_hw ?? prev?.frame_hw ?? null,
      }))
    }
    sock.onerror = () => setError('WebSocket error')
    sock.onclose = () => {
      ws.current = null
    }
    ws.current = sock
  }, [])

  const start = async () => {
    setError(null)
    setStarting(true)
    try {
      const res = await fetch(`/api/live/start?sport=swimming&standing=${standing}&engine=${engine}`, {
        method: 'POST',
      })
      const body = await res.json()
      if (!res.ok) throw new Error(body?.detail ?? 'start failed')
      openWs()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setStarting(false)
    }
  }

  // Attach to a swimming session that is already running (page reload, or it
  // crashed) so it can still be seen and stopped.
  useEffect(() => {
    fetch('/api/live/status')
      .then((r) => r.json())
      .then((d) => {
        if (d?.running && d.sport === 'swimming') openWs()
      })
      .catch(() => undefined)
  }, [openWs])

  const [stopping, setStopping] = useState(false)
  const stop = async () => {
    setStopping(true)
    closeWs()
    try {
      const res = await fetch('/api/live/stop', { method: 'POST' })
      if (!res.ok) setError(`Stop failed: HTTP ${res.status}`)
      else setError(null)
    } catch (e) {
      setError(`Stop failed: ${(e as Error).message}`)
    } finally {
      setStopping(false)
    }
    setSnap(null)
  }

  const otherSport = snap?.running && snap.sport && snap.sport !== 'swimming'
  const running = !!snap?.running && !otherSport
  useEffect(() => {
    onRunningChange?.(running || starting)
  }, [running, starting, onRunningChange])
  const fill = snap ? Math.round((snap.buffer_fill / Math.max(snap.window, 1)) * 100) : 0
  const stroke = snap?.stroke ? info(snap.stroke) : null

  let headline: { text: string; sub: string; color?: string }
  if (!running) headline = { text: '—', sub: 'Not started' }
  else if (snap?.status === 'no_person')
    headline = { text: 'No one detected', sub: 'Step back so your whole upper body is visible' }
  else if (snap?.status === 'arms_not_visible')
    headline = {
      text: 'Arms not visible · الدراعات مش باينة',
      sub: 'Step back until both arms and hands are in the frame',
    }
  else if (snap?.status === 'not_swimming')
    headline = {
      text: 'Not swimming · مش بتعوم',
      sub: `Arms barely moving (${(snap.arm_travel ?? 0).toFixed(1)} shoulder-widths in 2s, need ≥ 0.8)`,
      color: '#9aa4b2',
    }
  else if (snap?.status === 'uncertain')
    headline = {
      text: 'Uncertain',
      sub: `Best guess ${info(snap.prediction?.top1?.label ?? '').name} at ${((snap.confidence ?? 0) * 100).toFixed(0)}% (below ${60}%)`,
    }
  else if (stroke)
    headline = {
      text: `${stroke.name} · ${stroke.ar}`,
      sub: `${((snap?.confidence ?? 0) * 100).toFixed(0)}% confidence`,
      color: stroke.color,
    }
  else headline = { text: 'Watching…', sub: `Collecting ${snap?.window_seconds ?? 2}s of movement` }

  return (
    <section className="panel">
      <h2>Live stroke detection</h2>

      <div className="row" style={{ marginBottom: 12 }}>
        <button
          className="primary"
          onClick={start}
          // Only a CONFIRMED missing model blocks the button; the server still
          // refuses a start without it, and says why.
          disabled={starting || running || modelReady === false || !engine}
          title={engine ? undefined : 'Choose a pose model first (step 1)'}
        >
          {starting ? 'Starting… (~20s)' : running ? 'Running' : 'Start live detection'}
        </button>
        {/* Always clickable - it used to be disabled whenever the page thought
            nothing was running, which is exactly when a crashed session needs it. */}
        <button onClick={stop} disabled={stopping}>
          {stopping ? 'Stopping…' : 'Stop'}
        </button>
        <label className="small" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <input
            type="checkbox"
            checked={standing}
            disabled={running || starting}
            onChange={(e) => setStanding(e.target.checked)}
          />
          Standing simulation (rotate skeleton 90°)
        </label>
      </div>

      {modelReady === false && (
        <div className="notice warn" style={{ marginBottom: 12 }}>
          The swimming model is not installed, so live detection is unavailable.
        </div>
      )}
      {otherSport && (
        <div className="notice warn" style={{ marginBottom: 12 }}>
          A karate live session is using the camera. Stop it on the karate page first.
        </div>
      )}
      {(error || snap?.error) && (
        <div className="notice err" style={{ marginBottom: 12 }}>
          <strong>Live error</strong>
          <pre>{error ?? snap?.error}</pre>
        </div>
      )}

      <div className="grid2">
        <SkeletonCanvas
          players={running ? snap?.frame_players ?? snap?.latest_players ?? null : null}
          people={running ? snap?.frame_keypoints ?? snap?.latest_keypoints ?? null : null}
          frameHw={snap?.frame_hw ?? null}
          background={running ? snap?.latest_frame_jpeg ?? null : null}
          mirror={false}
        />
        <div>
          <div className="pred-top">
            <div className="small muted">Current stroke (last {snap?.window_seconds ?? 2}s)</div>
            <div className="pred-label" style={{ color: headline.color }}>
              {headline.text}
            </div>
            <div className="pred-conf">{headline.sub}</div>
          </div>

          {running && (
            <>
              <div className="small muted" style={{ marginBottom: 4 }}>
                Window {fill}% filled · people {snap?.people_now ?? 0}
                {snap?.inference_s != null && <> · infer {snap.inference_s}s</>}
                {snap?.lag_ms != null && <> · lag {snap.lag_ms} ms</>}
                {snap?.arm_travel != null && <> · arm travel {snap.arm_travel}</>}
              </div>
              <div className="bar-track" style={{ height: 6, marginBottom: 12 }}>
                <div className="bar-fill" style={{ width: `${fill}%` }} />
              </div>
              {(snap?.prediction?.topk ?? []).map((t, i) => {
                const I = info(t.label)
                return (
                  <div className="bar-row" key={t.label}>
                    <span className="bar-rank">{i + 1}</span>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: `${t.score * 100}%`, background: I.color }}
                      />
                      <span className="bar-name">
                        {I.name} · {I.ar}
                      </span>
                    </div>
                    <span className="bar-val">{(t.score * 100).toFixed(0)}%</span>
                  </div>
                )
              })}
            </>
          )}
        </div>
      </div>

      <div className="notice small" style={{ marginTop: 12 }}>
        Act out a stroke facing or side-on to the camera, arms fully in frame. With
        <strong> standing simulation</strong> on, your upright skeleton is turned 90° so it
        looks like a swimmer lying in the water - the orientation the model was trained on.
        This is a test aid; a standing imitation is not real swimming.
      </div>
    </section>
  )
}
