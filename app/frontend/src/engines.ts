/**
 * One place that describes each pose model, so no screen hard-codes OpenPose.
 *
 * All three engines are stored in OpenPose's JSON schema (BODY_25 slots) so the
 * pipeline is shared; what differs is what each engine natively produces.
 */
export interface EngineMeta {
  id: string
  label: string
  native: string // what the model outputs itself
  feet: boolean // toe/heel joints available
  boxes: string // where person boxes come from
  tracker: string // who assigns identities across frames
  toBody25: string // how its output is placed into the BODY_25 slots
}

export const ENGINES: Record<string, EngineMeta> = {
  openpose: {
    id: 'openpose',
    label: 'OpenPose BODY_25',
    native: 'BODY_25 (25 joints incl. feet)',
    feet: true,
    boxes: 'none from the model - drawn from the joint extent',
    tracker: 'nearest-centroid (app)',
    toBody25: 'native - no mapping',
  },
  rtmpose: {
    id: 'rtmpose',
    label: 'RTMPose-m + RTMDet-m',
    native: 'Halpe26 (COCO-17 + feet), boxes from RTMDet',
    feet: true,
    boxes: 'RTMDet person detector',
    tracker: 'nearest-centroid (app)',
    toBody25: 'COCO-17 + feet copied to their BODY_25 slots; Neck / MidHip = midpoints',
  },
  yolo11: {
    id: 'yolo11',
    label: 'YOLO11m-pose',
    native: 'COCO-17 (no feet) + boxes + ByteTrack ids, one pass',
    feet: false,
    boxes: 'YOLO11 (same pass as the pose)',
    tracker: 'ByteTrack (YOLO11)',
    toBody25: 'COCO-17 copied to their BODY_25 slots; Neck / MidHip = midpoints; feet empty',
  },
}

export const engineMeta = (id?: string | null): EngineMeta => ENGINES[id ?? 'openpose'] ?? ENGINES.openpose
