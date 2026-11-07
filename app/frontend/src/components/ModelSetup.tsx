import type { PoseEngine } from './UploadPanel'

/**
 * Step 1 on each page: choose the pose model BEFORE live detection or upload.
 * One choice drives both. The action-recognition model is fixed per page and
 * shown here so it is clear which classifier runs after the pose model.
 */
export function ModelSetup({
  engines,
  value,
  onChange,
  locked,
  actionModel,
}: {
  engines?: PoseEngine[]
  value: string | null
  onChange: (id: string) => void
  /** true while a live session or an analysis is running */
  locked?: boolean
  actionModel: { name: string; detail: string }
}) {
  return (
    <section className="panel">
      <h2>1 · Choose the model</h2>
      <div className="small muted" style={{ marginBottom: 8 }}>
        Pose model (finds the people and their joints) - used by both live detection and
        video upload:
      </div>
      {!engines ? (
        <div className="small muted">Loading available models…</div>
      ) : (
        <div className="row">
          {engines.map((e) => (
            <button
              key={e.id}
              className={`toggle ${value === e.id ? 'active' : ''}`}
              disabled={locked || !e.ready}
              title={e.ready ? `${e.keypoints} · ${e.license}` : 'not installed'}
              onClick={() => onChange(e.id)}
            >
              {e.label}
            </button>
          ))}
        </div>
      )}
      {value ? (
        <div className="small muted" style={{ marginTop: 8 }}>
          Keypoints: {engines?.find((e) => e.id === value)?.keypoints} · License:{' '}
          {engines?.find((e) => e.id === value)?.license}
          {locked && ' · locked while running'}
        </div>
      ) : (
        <div className="notice warn small" style={{ marginTop: 10 }}>
          Pick a pose model first - live detection and upload stay disabled until you do.
        </div>
      )}
      <div className="small" style={{ marginTop: 12 }}>
        <span className="muted">Action recognition (fixed for this page): </span>
        <strong>{actionModel.name}</strong>
        <span className="muted"> - {actionModel.detail}</span>
      </div>
    </section>
  )
}
