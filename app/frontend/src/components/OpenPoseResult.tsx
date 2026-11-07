import { useState } from 'react'
import { inputVideoUrl, openposeVideoUrl, skeletonJsonUrl } from '../api'
import { engineMeta } from '../engines'
import type { ResultResponse } from '../types'

type View = 'original' | 'skeleton'

export function OpenPoseResult({ result }: { result: ResultResponse }) {
  const [view, setView] = useState<View>('skeleton')
  const op = result.openpose
  const meta = result.video_meta
  const sk = result.skeleton
  const jobId = result.job_id

  const overlayReady = op?.mp4_written === true
  const em = engineMeta(result.pose_engine ?? op?.engine)

  return (
    <section className="panel">
      <h2>Pose result · {em.label}</h2>

      <div className="row" style={{ marginBottom: 12 }}>
        <button
          className={`toggle ${view === 'original' ? 'active' : ''}`}
          onClick={() => setView('original')}
        >
          Original
        </button>
        <button
          className={`toggle ${view === 'skeleton' ? 'active' : ''}`}
          onClick={() => setView('skeleton')}
          disabled={!overlayReady}
        >
          {op?.players ? 'Players' : 'Skeleton'} ({em.label})
        </button>
      </div>

      {view === 'original' ? (
        <video key="orig" src={inputVideoUrl(jobId)} controls loop playsInline />
      ) : overlayReady ? (
        <video key="ovl" src={openposeVideoUrl(jobId)} controls loop playsInline />
      ) : (
        <div className="notice warn">
          The skeleton overlay video is not available as browser-playable mp4.
          {op?.video_written
            ? ` ${em.label} wrote an AVI but the ffmpeg transcode did not produce an mp4.`
            : ` ${em.label} did not write an overlay video.`}
        </div>
      )}

      <div className="grid2" style={{ marginTop: 16 }}>
        <dl className="stats">
          <dt>Frames processed</dt>
          <dd>{op?.json_frame_count ?? '—'}</dd>

          <dt>Person detections</dt>
          <dd>{sk?.total_person_detections ?? '—'}</dd>

          <dt>Avg people / frame</dt>
          <dd>{sk?.avg_people_per_frame ?? '—'}</dd>

          <dt>Max people in a frame</dt>
          <dd>{sk?.max_people_in_a_frame ?? '—'}</dd>

          <dt>Tracks found</dt>
          <dd>{sk?.track_count ?? '—'} · {sk?.tracker ?? em.tracker}</dd>

          <dt>Players</dt>
          <dd>
            {op?.players
              ? op.players.labels
                  .map((l, i) => `${l} = track ${op.players!.track_ids[i]} (${op.players!.frames_present[l]} frames)`)
                  .join(' · ')
              : '—'}
          </dd>

          <dt>Processing time</dt>
          <dd>
            {op ? `${op.duration_s}s (${op.seconds_per_frame}s / frame)` : '—'}
          </dd>
        </dl>

        <dl className="stats">
          <dt>Model output</dt>
          <dd>{em.native}</dd>

          <dt>Stored as</dt>
          <dd>BODY_25 slots (OpenPose JSON schema) - {em.toBody25}</dd>

          <dt>Person boxes</dt>
          <dd>{em.boxes}</dd>

          <dt>Input size</dt>
          <dd>{op?.net_resolution ?? '—'}</dd>

          <dt>Device</dt>
          <dd>{op?.device ?? '—'}</dd>

          <dt>JSON schema version</dt>
          <dd>{sk?.json_versions?.join(', ') || '—'}</dd>

          <dt>Source resolution</dt>
          <dd>{meta ? `${meta.width}×${meta.height} @ ${meta.fps}fps` : '—'}</dd>

          <dt>JSON output</dt>
          <dd className="small">{op?.json_dir ?? '—'}</dd>
        </dl>
      </div>

      {meta?.analysis_frame_limit && meta.frame_count && meta.frame_count > meta.analysis_frame_limit && (
        <div className="notice warn" style={{ marginTop: 14 }}>
          Analysed the first <strong>{meta.analysis_frame_limit}</strong> of{' '}
          <strong>{meta.frame_count}</strong> frames
          {meta.fps ? (
            <>
              {' '}(<strong>{(meta.analysis_frame_limit / meta.fps).toFixed(0)}s</strong> of{' '}
              <strong>{(meta.frame_count / meta.fps).toFixed(0)}s</strong>)
            </>
          ) : null}
          , because the frame cap is set to <strong>{meta.analysis_frame_limit}</strong>.
          Running on {op?.device ?? 'unknown device'} at ~{op?.seconds_per_frame}s per
          frame. Set <code>OPENPOSE_MAX_FRAMES_GPU=0</code> in <code>.env</code> (frame cap for every pose model) to
          process the whole clip
          {meta.fps && op?.seconds_per_frame ? (
            <> (about{' '}
              <strong>
                {((meta.frame_count * op.seconds_per_frame) / 60).toFixed(0)} min
              </strong>{' '}
              at this rate)
            </>
          ) : null}
          .
        </div>
      )}

      <div className="row" style={{ marginTop: 14 }}>
        <a className="link" href={skeletonJsonUrl(jobId)} target="_blank" rel="noreferrer">
          <button>View Skeleton JSON</button>
        </a>
        <a className="link" href={openposeVideoUrl(jobId)} target="_blank" rel="noreferrer">
          <button>Download overlay video</button>
        </a>
      </div>

      {op?.command && (
        <details style={{ marginTop: 14 }}>
          <summary>Pose command used ({em.label})</summary>
          <pre className="json">{op.command.join(' \\\n  ')}</pre>
          <div className="small muted">cwd: {op.cwd}</div>
        </details>
      )}

      {sk && (
        <details style={{ marginTop: 10 }}>
          <summary>Skeleton structure summary (not the full JSON)</summary>
          <pre className="json">{JSON.stringify(
            {
              keypoint_format: sk.keypoint_format,
              num_keypoints: sk.num_keypoints,
              per_person_array_shape: sk.per_person_array_shape,
              frame_count: sk.frame_count,
              tracks: sk.tracks,
              sample_frames: sk.sample_frames,
            },
            null,
            2,
          )}</pre>
        </details>
      )}
    </section>
  )
}
