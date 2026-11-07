import { useCallback, useEffect, useRef, useState } from 'react'
import { SkeletonCanvas, type Keypoint, type LivePlayer } from './SkeletonCanvas'

interface Prediction {
  rank: number
  class_index: number
  label: string
  score: number
}

interface LiveSnapshot {
  running: boolean
  frames_processed: number
  people_now: number
  buffer_fill: number
  window: number
  inference_s: number | null
  error: string | null
  openpose_device: string | null
  mmaction_device: string | null
  frame_hw?: [number, number] | null
  latest_frame_jpeg?: string | null
  latest_keypoints?: Keypoint[][] | null
  frame_keypoints?: Keypoint[][] | null
  frame_players?: LivePlayer[] | null
  latest_players?: LivePlayer[] | null
  status?: 'warming' | 'no_person' | 'ok'
  sport?: string
  lag_ms?: number | null
  body_coverage?: number | null
  window_seconds?: number
  prediction: { top1: Prediction | null; topk: Prediction[]; num_clips: number } | null
}

interface StartResponse {
  running: boolean
  openpose_device: string | null
  mmaction_device: string | null
  gpu_build: boolean
  warning: string | null
}

/** engine: the pose model chosen at the top of the page (null = not chosen). */
export function LiveView({
  engine,
  onRunningChange,
}: {
  engine: string | null
  onRunningChange?: (running: boolean) => void
}) {
  const [snap, setSnap] = useState<LiveSnapshot | null>(null)
  const [starting, setStarting] = useState(false)
  const [startWarning, setStartWarning] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [fps, setFps] = useState(0)

  const ws = useRef<WebSocket | null>(null)
  const frameTimes = useRef<number[]>([])
  const lastFrameNo = useRef(-1)

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
      const data = JSON.parse(ev.data) as LiveSnapshot
      // Keep each image paired with the keypoints of that same frame: the
      // socket only ships a JPEG when a new one exists.
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
      if (data.frames_processed !== lastFrameNo.current) {
        lastFrameNo.current = data.frames_processed
        const now = performance.now()
        frameTimes.current.push(now)
        frameTimes.current = frameTimes.current.filter((t) => now - t < 3000)
        if (frameTimes.current.length > 1) {
          const span = (now - frameTimes.current[0]) / 1000
          setFps(span > 0 ? (frameTimes.current.length - 1) / span : 0)
        }
      }
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
      // OpenPose must claim the camera FIRST. If the browser opens it first the
      // device can refuse OpenPose a handle, and then there is no skeleton at
      // all - a pretty preview with a dead pipeline. Detection takes priority
      // over the preview.
      const res = await fetch(`/api/live/start?sport=karate&engine=${engine}`, { method: 'POST' })
      const body = await res.json()
      if (!res.ok) throw new Error(body?.detail ?? 'start failed')
      setStartWarning((body as StartResponse).warning)
      frameTimes.current = []
      lastFrameNo.current = -1
      openWs()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setStarting(false)
    }
  }

  // A session started before a page reload (or one that crashed) must still
  // be visible and stoppable: attach to it on mount.
  useEffect(() => {
    fetch('/api/live/status')
      .then((r) => r.json())
      .then((d) => {
        if (d?.running && (d.sport ?? 'karate') === 'karate') openWs()
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
    setFps(0)
  }

  const running = snap?.running ?? false
  // Lets the page lock the model choice while a session is live.
  useEffect(() => {
    onRunningChange?.(running || starting)
  }, [running, starting, onRunningChange])
  const top1 = snap?.prediction?.top1 ?? null
  const bufPct = snap ? Math.round((snap.buffer_fill / Math.max(snap.window, 1)) * 100) : 0
  const warming = starting || (running && (snap?.buffer_fill ?? 0) < (snap?.window ?? 100))

  return (
    <section className="panel">
      <h2>Live detection</h2>

      <div className="row" style={{ marginBottom: 14 }}>
        <button
          className="primary"
          onClick={start}
          disabled={starting || running || !engine}
          title={engine ? undefined : 'Choose a pose model first (step 1)'}
        >
          {starting ? 'Starting… (~20s warm-up)' : running ? 'Running' : 'Start live detection'}
        </button>
        {/* Always clickable: Stop must work after a crash or a page reload too,
            and stopping when nothing runs is harmless. */}
        <button onClick={stop} disabled={stopping}>
          {stopping ? 'Stopping…' : 'Stop'}
        </button>
        {running && (
          <span className="file-meta">
            <strong>{fps.toFixed(1)}</strong> fps · frame{' '}
            <strong>{snap?.frames_processed ?? 0}</strong> · people{' '}
            <strong>{snap?.people_now ?? 0}</strong>
            {snap?.inference_s != null && (
              <>
                {' · '}infer <strong>{snap.inference_s}s</strong>
              </>
            )}
            {snap?.lag_ms != null && (
              <>
                {' · '}camera lag <strong>{snap.lag_ms} ms</strong>
              </>
            )}
          </span>
        )}
      </div>

      {running && snap?.body_coverage != null && snap.body_coverage < 0.7 && (
        <div className="notice warn" style={{ marginBottom: 12 }}>
          Only {Math.round(snap.body_coverage * 100)}% of the body joints are visible. NTU60
          was trained on whole bodies - step back 2–3 m so you are seen head to feet.
        </div>
      )}

      {starting && (
        <div className="notice" style={{ marginBottom: 12 }}>
          Starting the pose model and loading ST-GCN — this takes about 20 seconds. The
          button stays disabled until it is ready; clicking again is not needed.
        </div>
      )}

      {startWarning && (
        <div className="notice warn" style={{ marginBottom: 12 }}>
          {startWarning}
        </div>
      )}

      {(error || snap?.error) && (
        <div className="notice err" style={{ marginBottom: 12 }}>
          <strong>Live pipeline error</strong>
          <pre>{error ?? snap?.error}</pre>
        </div>
      )}

      {/* ---------------- side by side: original | skeleton ---------------- */}
      <div className="grid2">
        <div>
          <div className="small muted" style={{ marginBottom: 6 }}>
            Original camera
          </div>
          {snap?.latest_frame_jpeg ? (
            <img
              src={`data:image/jpeg;base64,${snap.latest_frame_jpeg}`}
              alt="live camera frame"
              style={{ width: '100%', borderRadius: 8, background: '#000', display: 'block' }}
            />
          ) : (
            <div className="notice small">
              {running ? 'Waiting for the first frame…' : 'Not started.'}
            </div>
          )}
        </div>

        <div>
          <div className="small muted" style={{ marginBottom: 6 }}>
            Players (Player 1 / Player 2: the most-moving people) on the same frame
          </div>
          {/* mirror=false so the skeleton lines up with the frame on the left,
              which OpenPose writes unmirrored. */}
          <SkeletonCanvas
            players={snap?.frame_players ?? snap?.latest_players ?? null}
            people={snap?.frame_keypoints ?? snap?.latest_keypoints ?? null}
            frameHw={snap?.frame_hw ?? null}
            background={snap?.latest_frame_jpeg ?? null}
            mirror={false}
          />
        </div>
      </div>

      <div className="notice small" style={{ marginTop: 10 }}>
        The webcam is opened by the pose model on the backend — this device does not allow a
        second reader, so the browser cannot open the camera itself. Left is the clean
        frame; right is the <strong>same frame</strong> with the players' boxes and skeletons drawn
        over it, so the joints sit on your body. The overlay is drawn in the browser
        from the streamed keypoints, not burnt into the image, which is why the left
        pane stays clean.
      </div>

      {/* ---------------------------- prediction ---------------------------- */}
      {running && (
        <>
          <div style={{ marginTop: 16 }}>
            <div className="small muted" style={{ marginBottom: 4 }}>
              ST-GCN sliding window: last {snap?.window_seconds ?? 2.5}s ({snap?.buffer_fill ?? 0}%
              filled)
            </div>
            <div className="bar-track" style={{ height: 8 }}>
              <div className="bar-fill" style={{ width: `${bufPct}%` }} />
            </div>
          </div>

          {snap?.status === 'no_person' ? (
            <div className="notice warn" style={{ marginTop: 14 }}>
              <strong>No one in front of the camera.</strong> Stand where your whole body is
              visible - classification resumes as soon as you are detected.
            </div>
          ) : top1 ? (
            <>
              <div className="pred-top" style={{ marginTop: 14 }}>
                <div className="small muted">Live classification (rolling window)</div>
                <div className="pred-label">{top1.label}</div>
                <div className="pred-conf">{(top1.score * 100).toFixed(1)}%</div>
              </div>
              {snap?.prediction?.topk.slice(0, 5).map((t) => (
                <div className="bar-row" key={t.rank}>
                  <span className="bar-rank">{t.rank}.</span>
                  <span className="bar-track">
                    <span className="bar-fill" style={{ width: `${t.score * 100}%` }} />
                    <span className="bar-name">{t.label}</span>
                  </span>
                  <span className="bar-val">{t.score.toFixed(3)}</span>
                </div>
              ))}
            </>
          ) : (
            <div className="notice small" style={{ marginTop: 14 }}>
              {warming
                ? `Filling the ${snap?.window ?? 100}-frame window before the first prediction…`
                : 'Waiting for a prediction…'}
            </div>
          )}

          <div className="grid2" style={{ marginTop: 14 }}>
            <dl className="stats">
              <dt>Pose model</dt>
              <dd>{snap?.openpose_device ?? '—'}</dd>
              <dt>MMAction2 device</dt>
              <dd>{snap?.mmaction_device ?? '—'}</dd>
            </dl>
            <dl className="stats">
              <dt>Camera frame</dt>
              <dd>
                {snap?.frame_hw ? `${snap.frame_hw[1]}×${snap.frame_hw[0]}` : '—'}
              </dd>
              <dt>num_clips</dt>
              <dd>
                {snap?.prediction?.num_clips ?? '—'}
                <span className="muted"> (1 = low latency)</span>
              </dd>
            </dl>
          </div>

          <div className="notice warn small" style={{ marginTop: 12 }}>
            Labels come from <strong>NTU RGB+D 60</strong>, which contains no karate
            classes. These are out-of-domain predictions, not karate technique labels.
          </div>
        </>
      )}
    </section>
  )
}
