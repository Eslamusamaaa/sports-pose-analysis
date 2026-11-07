import { useEffect, useRef, useState } from 'react'
import { inputVideoUrl, openposeVideoUrl } from '../api'
import { engineMeta } from '../engines'
import type { ResultResponse, SwimAnalysis } from '../types'

export const STROKE_INFO: Record<string, { name: string; ar: string; color: string }> = {
  freestyle: { name: 'Freestyle', ar: 'حرة', color: '#4da3ff' },
  backstroke: { name: 'Backstroke', ar: 'ظهر', color: '#3ddc97' },
  breaststroke: { name: 'Breaststroke', ar: 'صدر', color: '#ffb74d' },
  butterfly: { name: 'Butterfly', ar: 'فراشة', color: '#c792ea' },
  uncertain: { name: 'Uncertain', ar: 'غير متأكد', color: '#6b7280' },
  not_visible: { name: 'Not visible', ar: 'السباح مش ظاهر', color: '#2a2f38' },
}

const info = (k: string) => STROKE_INFO[k] ?? { name: k, ar: '', color: '#888' }

function fmt(s: number) {
  const m = Math.floor(s / 60)
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')}`
}

/** Colored strip of stroke segments under the video; click to seek. */
function StrokeStrip({
  swim,
  duration,
  current,
  onSeek,
}: {
  swim: SwimAnalysis
  duration: number
  current: number
  onSeek: (t: number) => void
}) {
  const last = swim.segments[swim.segments.length - 1]
  const total = Math.max(duration, last?.end_s ?? 0, 0.001)
  return (
    <div>
      <div className="swim-strip">
        {swim.segments.map((seg, i) => {
          const I = info(seg.stroke)
          return (
            <button
              key={i}
              className="swim-seg"
              title={`${I.name} ${fmt(seg.start_s)}–${fmt(seg.end_s)} · ${(seg.mean_confidence * 100).toFixed(0)}%`}
              style={{
                left: `${(seg.start_s / total) * 100}%`,
                width: `${((seg.end_s - seg.start_s) / total) * 100}%`,
                background: I.color,
              }}
              onClick={() => onSeek(seg.start_s)}
            />
          )
        })}
        <div className="swim-cursor" style={{ left: `${(current / total) * 100}%` }} />
      </div>
      <div className="row small muted" style={{ marginTop: 8, gap: 14 }}>
        {Object.entries(STROKE_INFO).map(([k, v]) => (
          <span key={k}>
            <span className="swim-dot" style={{ background: v.color }} /> {v.name} ({v.ar})
          </span>
        ))}
      </div>
    </div>
  )
}

export function SwimResult({ result }: { result: ResultResponse }) {
  const swim = result.mmaction?.swimming
  const videoRef = useRef<HTMLVideoElement>(null)
  const [current, setCurrent] = useState(0)
  const [duration, setDuration] = useState(result.video_meta?.duration_s ?? 0)
  const [skeleton, setSkeleton] = useState(true)
  const hasOverlay = !!result.outputs?.openpose_video_mp4?.exists

  useEffect(() => setSkeleton(hasOverlay), [hasOverlay, result.job_id])

  if (!swim) return null

  const seek = (t: number) => {
    const v = videoRef.current
    if (v) {
      v.currentTime = t
      void v.play()
    }
  }
  const now = swim.windows.find((w) => current >= w.start_s && current < w.end_s)
  const dom = swim.dominant_stroke ? info(swim.dominant_stroke) : null

  return (
    <>
      <section className="panel">
        <h2>
          Stroke recognition <span className="tag">SYNTHETIC-TRAINED</span>{' '}
          <span className="small muted">
            keypoints: {engineMeta(result.pose_engine).label}
          </span>
        </h2>

        <div className="grid2">
          <div>
            <video
              ref={videoRef}
              key={`${result.job_id}-${skeleton}`}
              src={skeleton ? openposeVideoUrl(result.job_id) : inputVideoUrl(result.job_id)}
              controls
              onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
              onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || duration)}
            />
            <div className="row" style={{ marginTop: 8 }}>
              <button
                className={`toggle ${skeleton ? 'active' : ''}`}
                disabled={!hasOverlay}
                onClick={() => setSkeleton(true)}
              >
                Skeleton
              </button>
              <button className={`toggle ${!skeleton ? 'active' : ''}`} onClick={() => setSkeleton(false)}>
                Original
              </button>
            </div>
            <div style={{ marginTop: 12 }}>
              <StrokeStrip swim={swim} duration={duration} current={current} onSeek={seek} />
            </div>
          </div>

          <div>
            <div className="pred-top">
              <div className="small muted">Main stroke</div>
              <div className="pred-label" style={{ color: dom?.color }}>
                {dom ? `${dom.name} · ${dom.ar}` : 'Not determined'}
              </div>
              <div className="pred-conf">
                {swim.visible_frames != null
                  ? `swimmer seen in ${swim.visible_frames}/${swim.total_frames} frames · ${
                      swim.num_windows - (swim.not_visible_windows ?? 0)
                    }/${swim.num_windows} windows analysed`
                  : `${swim.num_windows - swim.uncertain_windows}/${swim.num_windows} windows confident`}
              </div>
            </div>
            {swim.visible_frames != null && swim.visible_frames < 0.5 * (swim.total_frames ?? 0) && (
              <div className="notice warn" style={{ marginBottom: 12 }}>
                {engineMeta(result.pose_engine).label} found the swimmer in only{' '}
                {Math.round((100 * swim.visible_frames) / Math.max(swim.total_frames ?? 1, 1))}% of
                frames, so this result rests on little data.
              </div>
            )}

            <div className="small muted" style={{ marginBottom: 6 }}>
              Now at {fmt(current)}:{' '}
              <strong style={{ color: now ? info(now.stroke).color : undefined }}>
                {now ? `${info(now.stroke).name} (${(now.confidence * 100).toFixed(0)}%)` : '—'}
              </strong>
            </div>

            {swim.distribution.map((d, i) => {
              const I = info(d.stroke)
              return (
                <div className="bar-row" key={d.stroke}>
                  <span className="bar-rank">{i + 1}</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${d.share * 100}%`, background: I.color }} />
                    <span className="bar-name">
                      {I.name} · {I.ar}
                    </span>
                  </div>
                  <span className="bar-val">{(d.share * 100).toFixed(0)}%</span>
                </div>
              )
            })}
            <div className="small muted" style={{ marginTop: 6 }}>
              Share of confident {swim.window_seconds}s windows (step {swim.stride_seconds}s).
              Video {swim.input_fps.toFixed(1)} fps → model {swim.model_fps} fps.
            </div>
          </div>
        </div>

        <div className="notice warn" style={{ marginTop: 14 }}>
          {swim.domain_warning}
        </div>
      </section>

      <section className="panel">
        <h2>Coaching feedback</h2>
        {swim.coaching.tips.length === 0 ? (
          <div className="notice">
            <strong>Coming next.</strong> {swim.coaching.message}
            <div className="small muted" style={{ marginTop: 6 }}>
              Planned: arm recovery height, head position, kick depth &amp; rate, left/right
              symmetry, stroke rate - each tip linked to the moment in the video.
            </div>
          </div>
        ) : (
          <ul className="swim-tips">
            {swim.coaching.tips.map((t, i) => (
              <li key={i} className={t.severity === 'warn' ? 'warn' : ''}>
                {t.start_s !== undefined && (
                  <button onClick={() => seek(t.start_s!)}>{fmt(t.start_s)}</button>
                )}{' '}
                {t.message}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel">
        <h2>Segments</h2>
        <table className="kv">
          <thead>
            <tr>
              <th>From</th>
              <th>To</th>
              <th>Stroke</th>
              <th>Confidence</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {swim.segments.map((seg, i) => {
              const I = info(seg.stroke)
              return (
                <tr key={i}>
                  <td className="mono">{fmt(seg.start_s)}</td>
                  <td className="mono">{fmt(seg.end_s)}</td>
                  <td>
                    <span className="swim-dot" style={{ background: I.color }} /> {I.name} ({I.ar})
                  </td>
                  <td className="mono">{(seg.mean_confidence * 100).toFixed(0)}%</td>
                  <td>
                    <button onClick={() => seek(seg.start_s)}>Play</button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </section>
    </>
  )
}
