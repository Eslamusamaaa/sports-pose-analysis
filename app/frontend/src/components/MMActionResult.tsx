import { engineMeta } from '../engines'
import type { ConversionInfo, MMActionInfo, ResultResponse, StageInfo } from '../types'

function ConversionDetails({ conv, engine }: { conv: ConversionInfo; engine?: string }) {
  const em = engineMeta(engine)
  return (
    <details style={{ marginTop: 14 }}>
      <summary>
        Keypoint conversion: {conv.source_format} ({conv.source_num_keypoints}) →{' '}
        {conv.target_format} ({conv.target_num_keypoints})
      </summary>
      <p className="small muted" style={{ marginTop: 0 }}>
        Pose model: <strong>{em.label}</strong> - outputs {em.native}. Stored in OpenPose's
        BODY_25 slots ({em.toBody25}). The action model's graph is{' '}
        <code>layout='{conv.target_layout}'</code> = {conv.target_num_keypoints} nodes, so this
        step reads the {conv.target_num_keypoints} COCO joints back out of those slots - a pure
        subset + reorder, no joint is interpolated or invented.{' '}
        {em.id === 'openpose'
          ? `${conv.dropped_count} BODY_25 joints (Neck, MidHip, feet) are discarded.`
          : em.feet
            ? 'Neck / MidHip (synthesised midpoints) and the feet are not used by the action model.'
            : 'Neck / MidHip (synthesised midpoints) are not used; this model has no feet to drop.'}
      </p>
      <div className="small muted" style={{ marginBottom: 8 }}>
        {conv.tensor_shape_note} · coordinates: {conv.coordinate_space}
      </div>
      <table className="kv">
        <thead>
          <tr>
            <th>COCO idx</th>
            <th>COCO name</th>
            <th>← BODY_25 idx</th>
            <th>BODY_25 name</th>
          </tr>
        </thead>
        <tbody>
          {conv.mapping.map((m) => (
            <tr key={m.coco_index}>
              <td className="mono">{m.coco_index}</td>
              <td>{m.coco_name}</td>
              <td className="mono">{m.body25_index}</td>
              <td>{m.body25_name}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small muted">
        BODY_25 slots not sent to the action model:{' '}
        {conv.dropped_body25_joints.map((d) => `${d.body25_index} ${d.body25_name}`).join(', ')}
      </p>
    </details>
  )
}

function TopK({ mm }: { mm: MMActionInfo }) {
  const max = Math.max(...mm.topk.map((t) => t.score), 0.0001)
  return (
    <>
      {mm.top1 && (
        <div className="pred-top">
          <div className="small muted">Top prediction</div>
          <div className="pred-label">{mm.top1.label}</div>
          <div className="pred-conf">
            confidence {(mm.top1.score * 100).toFixed(1)}% · class index {mm.top1.class_index} of{' '}
            {mm.num_classes}
          </div>
        </div>
      )}
      {mm.topk.map((t) => (
        <div className="bar-row" key={t.rank}>
          <span className="bar-rank">{t.rank}.</span>
          <span className="bar-track">
            <span className="bar-fill" style={{ width: `${(t.score / max) * 100}%` }} />
            <span className="bar-name">{t.label}</span>
          </span>
          <span className="bar-val">{t.score.toFixed(4)}</span>
        </div>
      ))}
    </>
  )
}

interface Props {
  result: ResultResponse
  mmStage: StageInfo | undefined
}

export function MMActionResult({ result, mmStage }: Props) {
  const mm = result.mmaction
  const info = result.model_info
  const conv = result.conversion

  return (
    <section className="panel">
      <h2>MMAction2 result</h2>

      <div className="grid2" style={{ marginBottom: 14 }}>
        <dl className="stats">
          <dt>Model</dt>
          <dd>{info.model_name}</dd>
          <dt>Dataset</dt>
          <dd>{info.dataset}</dd>
          <dt>Status</dt>
          <dd>
            <span className="tag">{info.status_banner}</span>
          </dd>
          <dt>Classes</dt>
          <dd>{mm?.num_classes ?? 60}</dd>
        </dl>
        <dl className="stats">
          <dt>Keypoint layout</dt>
          <dd>
            {mm?.graph_layout ?? info.keypoint_layout} · {mm?.graph_num_node ?? 17} nodes
          </dd>
          <dt>Sent to model</dt>
          <dd>
            {mm
              ? `${mm.num_persons_sent} persons × ${mm.num_frames_sent} frames × ${mm.num_keypoints_sent} kpts`
              : '—'}
          </dd>
          <dt>Device</dt>
          <dd>
            {mm ? `${mm.device_used}` : '—'}
            {mm && mm.device_requested !== mm.device_used && (
              <span className="muted"> (requested {mm.device_requested})</span>
            )}
          </dd>
          <dt>Inference time</dt>
          <dd>{mm ? `${mm.inference_s}s (load ${mm.model_load_s}s)` : '—'}</dd>
        </dl>
      </div>

      {mm ? (
        <>
          <div className="notice warn" style={{ marginBottom: 14 }}>
            {mm.domain_warning}
          </div>
          <TopK mm={mm} />
          {mm.note && (
            <div className="notice warn" style={{ marginTop: 12 }}>
              {mm.note}
            </div>
          )}
        </>
      ) : (
        <div className={`notice ${mmStage?.state === 'failed' ? 'err' : 'warn'}`}>
          <strong>
            {mmStage?.state === 'failed'
              ? 'MMAction2 failed.'
              : mmStage?.state === 'skipped'
                ? 'MMAction2 was skipped.'
                : 'MMAction2 has not produced a prediction yet.'}
          </strong>
          {result.openpose && (
            <div style={{ marginTop: 6 }}>
              Pose extraction ({engineMeta(result.pose_engine).label}) completed - its outputs above remain valid and usable.
            </div>
          )}
          {mmStage?.error && <pre>{mmStage.error}</pre>}
        </div>
      )}

      {conv && <ConversionDetails conv={conv} engine={result.pose_engine} />}

      <details style={{ marginTop: 10 }}>
        <summary>Checkpoint &amp; config</summary>
        <dl className="stats" style={{ marginTop: 8 }}>
          <dt>config</dt>
          <dd className="small">{info.config}</dd>
          <dt>checkpoint</dt>
          <dd className="small">{info.checkpoint}</dd>
        </dl>
        <div className="notice warn" style={{ marginTop: 10 }}>
          {info.disclaimer}
        </div>
      </details>
    </section>
  )
}
