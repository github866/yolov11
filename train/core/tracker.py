class YoloDinoTracker:
    def __init__(self, detector, dino, metric='cosine', thr=0.5):
        ...
    def update(self, frame):
        boxes, confs, cls = self.detector(frame)
        feats   = self.dino.extract(frame, boxes)
        tracks  = self._match_tracks(boxes, feats)
        return tracks
